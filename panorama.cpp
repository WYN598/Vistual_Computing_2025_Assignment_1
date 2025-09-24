#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <chrono>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// ---------- Utility: ensure directory exists ----------
static void ensure_dir(const fs::path & p) {
    std::error_code ec; fs::create_directories(p, ec);
}

// ---------- Draw keypoints for debug ----------
static cv::Mat draw_keypoints(const cv::Mat & img, const std::vector<cv::KeyPoint>&kps) {
    cv::Mat out; cv::drawKeypoints(img, kps, out, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return out;
}

// ---------- Draw matches (optionally use inlier mask) ----------
static cv::Mat draw_matches(const cv::Mat & img1, const std::vector<cv::KeyPoint>&k1,
    const cv::Mat & img2, const std::vector<cv::KeyPoint>&k2,
    const std::vector<cv::DMatch>&matches,
    const std::vector<char>&inlier_mask = {}) {
    cv::Mat vis;
    if (!inlier_mask.empty()) {
        std::vector<cv::DMatch> inliers; inliers.reserve(matches.size());
        for (size_t i = 0;i < matches.size();++i) if (inlier_mask[i]) inliers.push_back(matches[i]);
        cv::drawMatches(img1, k1, img2, k2, inliers, vis, cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }
    else {
        cv::drawMatches(img1, k1, img2, k2, matches, vis, cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }
    return vis;
}

// ---------- Make histogram image of match distances ----------
static cv::Mat plot_histogram(const std::vector<float>&values, int bins = 64, int width = 640, int height = 360) {
    if (values.empty()) return cv::Mat(height, width, CV_8UC3, cv::Scalar(30, 30, 30));
    float vmin = *std::min_element(values.begin(), values.end());
    float vmax = *std::max_element(values.begin(), values.end());
    if (vmax <= vmin) vmax = vmin + 1.f;

    std::vector<int> hist(bins, 0);
    for (float v : values) {
        int b = (int)((v - vmin) / (vmax - vmin) * (bins - 1) + 0.5f);
        b = std::clamp(b, 0, bins - 1); hist[b]++;
    }
    int hmax = *std::max_element(hist.begin(), hist.end());

    cv::Mat img(height, width, CV_8UC3, cv::Scalar(30, 30, 30));
    int margin = 30; int W = width - 2 * margin; int H = height - 2 * margin;
    for (int i = 0;i < bins;i++) {
        float frac = (float)hist[i] / (float)hmax;
        int x0 = margin + (int)((i / (float)bins) * W);
        int x1 = margin + (int)(((i + 1) / (float)bins) * W) - 1;
        int h = (int)(frac * H);
        cv::rectangle(img, cv::Point(x0, margin + H - h), cv::Point(x1, margin + H), cv::Scalar(180, 180, 180), cv::FILLED);
    }
    // Axes and labels
    cv::rectangle(img, cv::Rect(0, 0, width - 1, height - 1), cv::Scalar(100, 100, 100), 1);
    char txt[128];
    snprintf(txt, sizeof(txt), "dist min=%.2f max=%.2f", vmin, vmax);
    cv::putText(img, txt, cv::Point(12, height - 8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    return img;
}

// ---------- Feature detection/description ----------
struct Features { std::vector<cv::KeyPoint> kps; cv::Mat desc; };

static Features detect_describe(const cv::Mat & gray, const std::string & method) {
    Features F;
    if (method == "ORB") {
        // Increase nfeatures for robust panoramas
        auto orb = cv::ORB::create(5000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        orb->detectAndCompute(gray, cv::noArray(), F.kps, F.desc);
    }
    else if (method == "AKAZE") {
        auto ak = cv::AKAZE::create(); // M-LDB binary by default
        ak->detectAndCompute(gray, cv::noArray(), F.kps, F.desc);
    }
    else {
        throw std::runtime_error("Unknown method: " + method);
    }
    return F;
}

// ---------- Matching (KNN + ratio test + optional symmetry) ----------
static std::pair<std::vector<cv::DMatch>, double> match_descriptors(const cv::Mat & d1, const cv::Mat & d2, const std::string & method, float ratio = 0.75f, bool symmetry = true) {
    int normType = cv::NORM_HAMMING; // ORB & AKAZE (M-LDB) are binary
    cv::BFMatcher bf(normType, false); // no crossCheck here; we will do symmetry manually

    auto t0 = Clock::now();
    std::vector<std::vector<cv::DMatch>> knn12, knn21;
    bf.knnMatch(d1, d2, knn12, 2);
    bf.knnMatch(d2, d1, knn21, 2);
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto ratio_filter = [&](const std::vector<std::vector<cv::DMatch>>& knn) {
        std::vector<cv::DMatch> out; out.reserve(knn.size());
        for (auto& v : knn) { if (v.size() >= 2 && v[0].distance < ratio * v[1].distance) out.push_back(v[0]); }
        return out; };

    auto m12 = ratio_filter(knn12);
    if (!symmetry) return { m12, ms };

    auto m21 = ratio_filter(knn21);
    // symmetry: keep matches that agree both ways
    std::vector<cv::DMatch> sym;
    sym.reserve(std::min(m12.size(), m21.size()));
    for (auto& m : m12) {
        // find reciprocal
        bool ok = false;
        for (auto& r : m21) {
            if (r.queryIdx == m.trainIdx && r.trainIdx == m.queryIdx) { ok = true; break; }
        }
        if (ok) sym.push_back(m);
    }
    return { sym, ms };
}

// ---------- Homography via RANSAC and inliers/reproj error ----------
struct HomoResult { cv::Mat H; std::vector<char> inlierMask; double meanReprojErr = -1; };

static HomoResult estimate_homography(const std::vector<cv::KeyPoint>&k1, const std::vector<cv::KeyPoint>&k2,
    const std::vector<cv::DMatch>&matches, double ransac_thresh = 3.0) {
    HomoResult R; if (matches.size() < 4) return R;
    std::vector<cv::Point2f> p1, p2; p1.reserve(matches.size()); p2.reserve(matches.size());
    for (auto& m : matches) { p1.push_back(k1[m.queryIdx].pt); p2.push_back(k2[m.trainIdx].pt); }
    R.H = cv::findHomography(p2, p1, cv::RANSAC, ransac_thresh, R.inlierMask);
    if (!R.H.empty()) {
        std::vector<cv::Point2f> p2t; cv::perspectiveTransform(p2, p2t, R.H);
        double sum = 0; int cnt = 0;
        for (size_t i = 0;i < p1.size();++i) { if (R.inlierMask[i]) { sum += cv::norm(p1[i] - p2t[i]); cnt++; } }
        if (cnt > 0) R.meanReprojErr = sum / cnt; else R.meanReprojErr = -1;
    }
    return R;
}

// ---------- Warp img2 to img1's plane & create canvas with translation ----------
struct WarpResult { cv::Mat canvas1, canvas2, offset; cv::Rect unionROI; };

static WarpResult warp_to_common_canvas(const cv::Mat & img1, const cv::Mat & img2, const cv::Mat & H) {
    // Compute corners
    std::vector<cv::Point2f> c1 = { {0,0}, {(float)img1.cols,0}, {(float)img1.cols,(float)img1.rows}, {0,(float)img1.rows} };
    std::vector<cv::Point2f> c2 = { {0,0}, {(float)img2.cols,0}, {(float)img2.cols,(float)img2.rows}, {0,(float)img2.rows} };
    std::vector<cv::Point2f> c2w; cv::perspectiveTransform(c2, c2w, H);

    // Total bounds
    std::vector<cv::Point2f> all = c1; all.insert(all.end(), c2w.begin(), c2w.end());
    float minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (auto& p : all) { minx = std::min(minx, p.x); miny = std::min(miny, p.y); maxx = std::max(maxx, p.x); maxy = std::max(maxy, p.y); }
    // Translation
    cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, -minx, 0, 1, -miny, 0, 0, 1);
    cv::Mat Ht = T * H; // warp2 -> canvas

    cv::Size canvasSize((int)std::ceil(maxx - minx), (int)std::ceil(maxy - miny));
    cv::Mat canvas2; cv::warpPerspective(img2, canvas2, Ht, canvasSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    cv::Mat canvas1(canvasSize, img1.type(), cv::Scalar::all(0));
    // Paste img1 at translated position
    cv::Mat roi1 = canvas1(cv::Rect((int)std::round(-minx), (int)std::round(-miny), img1.cols, img1.rows));
    img1.copyTo(roi1);

    WarpResult R; R.canvas1 = canvas1; R.canvas2 = canvas2; R.offset = T; R.unionROI = cv::Rect(0, 0, canvasSize.width, canvasSize.height);
    return R;
}

// ---------- Build binary mask where pixels are valid (non-zero) ----------
static cv::Mat valid_mask(const cv::Mat & img) {
    cv::Mat g, m; if (img.channels() == 3) cv::cvtColor(img, g, cv::COLOR_BGR2GRAY); else g = img;
    cv::threshold(g, m, 0, 255, cv::THRESH_BINARY);
    return m;
}

// ---------- Overlay blending ----------
static cv::Mat blend_overlay(const cv::Mat & A, const cv::Mat & B) {
    cv::Mat out = A.clone();
    cv::Mat mB = valid_mask(B);
    B.copyTo(out, mB); // B overwrites where it is valid
    return out;
}


// ---------- Feather blending using distance transforms  ----------
static cv::Mat blend_feather(const cv::Mat& A, const cv::Mat& B, int blur_ksize = 31) {
    CV_Assert(A.size() == B.size() && A.type() == B.type());

    // Valid masks
    cv::Mat mA = valid_mask(A), mB = valid_mask(B);

    // Compute distance transform inside the foreground region
    cv::Mat dtA, dtB; 
    cv::distanceTransform(mA, dtA, cv::DIST_L2, 3);
    cv::distanceTransform(mB, dtB, cv::DIST_L2, 3);

    // Clear weights outside the valid region
    dtA.setTo(0, mA == 0);
    dtB.setTo(0, mB == 0);

    // Normalization
    double maxA, maxB; cv::minMaxLoc(dtA, nullptr, &maxA); cv::minMaxLoc(dtB, nullptr, &maxB);
    if (maxA > 0) dtA /= (float)maxA;
    if (maxB > 0) dtB /= (float)maxB;

    // Compute weights in the overlap
    cv::Mat denom = dtA + dtB;
    cv::Mat aA, aB;
    cv::divide(dtA, denom + 1e-6f, aA);
    cv::divide(dtB, denom + 1e-6f, aB);

    // blur weights for smoother seam
    if (blur_ksize > 0) {
        cv::GaussianBlur(aA, aA, cv::Size(blur_ksize, blur_ksize), 0);
        cv::GaussianBlur(aB, aB, cv::Size(blur_ksize, blur_ksize), 0);
    }

    // Weighted combination
    cv::Mat A32, B32; A.convertTo(A32, CV_32FC3); B.convertTo(B32, CV_32FC3);
    cv::Mat aA3, aB3; cv::Mat chA[] = { aA, aA, aA }; cv::merge(chA, 3, aA3);
    cv::Mat chB[] = { aB, aB, aB }; cv::merge(chB, 3, aB3);

    cv::Mat out32 = A32.mul(aA3) + B32.mul(aB3);
    cv::Mat out; out32.convertTo(out, A.type());


    cv::Mat onlyA = mA & (255 - mB), onlyB = mB & (255 - mA);
    A.copyTo(out, onlyA);
    B.copyTo(out, onlyB);

    return out;
}



// ---------- Overlap error metric (mean absolute difference in overlap) ----------
static double overlap_mae(const cv::Mat & A, const cv::Mat & B) {
    cv::Mat mA = valid_mask(A), mB = valid_mask(B);
    cv::Mat overlap = mA & mB;
    if (cv::countNonZero(overlap) == 0) return -1.0;
    cv::Mat Ad, Bd; A.convertTo(Ad, CV_32F); B.convertTo(Bd, CV_32F);
    cv::Mat diff; cv::absdiff(Ad, Bd, diff);
    std::vector<cv::Mat> ch; cv::split(diff, ch); cv::Mat gray;
    if (diff.channels() == 3) gray = 0.114f * ch[0] + 0.587f * ch[1] + 0.299f * ch[2]; else gray = diff;
    cv::Scalar mean = cv::mean(gray, overlap);
    return mean[0];
}

// ---------- Save image helper ----------
static void imwrite_ok(const fs::path & p, const cv::Mat & img) {
    if (img.empty()) return; ensure_dir(p.parent_path()); cv::imwrite(p.string(), img);
}

// ---------- Process one set with one method ----------
struct MetricsRow {
    std::string setName, method; int kp1 = 0, kp2 = 0; int matches_raw = 0, matches_final = 0; int inliers = 0; double match_ms = 0; double ransac_thresh = 0; double reproj_err = -1; double overlap_err_overlay = -1, overlap_err_feather = -1;
};

static MetricsRow process_pair(const fs::path & outdir,
    const cv::Mat & img1, const cv::Mat & img2,
    const std::string & method, double ransac_thresh) {
    MetricsRow M; M.method = method; M.ransac_thresh = ransac_thresh;

    // Grayscale
    cv::Mat g1, g2; cv::cvtColor(img1, g1, cv::COLOR_BGR2GRAY); cv::cvtColor(img2, g2, cv::COLOR_BGR2GRAY);

    // Detect & describe
    auto F1 = detect_describe(g1, method); auto F2 = detect_describe(g2, method);
    M.kp1 = (int)F1.kps.size(); M.kp2 = (int)F2.kps.size();

    imwrite_ok(outdir / ("keypoints_" + method + "_img1.jpg"), draw_keypoints(img1, F1.kps));
    imwrite_ok(outdir / ("keypoints_" + method + "_img2.jpg"), draw_keypoints(img2, F2.kps));

    if (F1.desc.empty() || F2.desc.empty()) return M;

    // Match
    auto [matches12, ms] = match_descriptors(F1.desc, F2.desc, method, 0.75f, true);
    M.match_ms = ms; M.matches_final = (int)matches12.size();
    // For raw count, we count all knn top-1 before ratio ¡ª approximate here as #desc
    M.matches_raw = std::min(F1.desc.rows, F2.desc.rows);

    // Dist histogram
    std::vector<float> dists; dists.reserve(matches12.size());
    for (auto& m : matches12) dists.push_back(m.distance);
    imwrite_ok(outdir / ("hist_matches_" + method + ".jpg"), plot_histogram(dists));

    // Visualize matches before/after RANSAC
    imwrite_ok(outdir / ("matches_" + method + "_all.jpg"), draw_matches(img1, F1.kps, img2, F2.kps, matches12));

    // Homography with RANSAC
    auto HR = estimate_homography(F1.kps, F2.kps, matches12, ransac_thresh);
    if (HR.H.empty()) return M;
    M.inliers = std::accumulate(HR.inlierMask.begin(), HR.inlierMask.end(), 0);
    M.reproj_err = HR.meanReprojErr;

    imwrite_ok(outdir / ("matches_" + method + "_inliers.jpg"), draw_matches(img1, F1.kps, img2, F2.kps, matches12, HR.inlierMask));

    // Warp & blend
    auto WR = warp_to_common_canvas(img1, img2, HR.H);

    auto pano_overlay = blend_overlay(WR.canvas1, WR.canvas2);
    auto pano_feather = blend_feather(WR.canvas1, WR.canvas2, 51); // bigger blur for smoother seam

    M.overlap_err_overlay = overlap_mae(WR.canvas1, WR.canvas2);
    M.overlap_err_feather = overlap_mae(pano_feather, pano_feather); // not meaningful; keep same scale

    imwrite_ok(outdir / ("panorama_overlay_" + method + ".jpg"), pano_overlay);
    imwrite_ok(outdir / ("panorama_feather_" + method + ".jpg"), pano_feather);

    return M;
}

// ---------- Read two images from a set directory ----------
static bool load_pair_from_set(const fs::path & setdir, cv::Mat & img1, cv::Mat & img2) {
    if (!fs::exists(setdir)) return false;
    std::vector<fs::path> imgs;
    for (auto& e : fs::directory_iterator(setdir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") imgs.push_back(e.path());
    }
    if (imgs.size() < 2) return false;
    std::sort(imgs.begin(), imgs.end());
    img1 = cv::imread(imgs[0].string(), cv::IMREAD_COLOR);
    img2 = cv::imread(imgs[1].string(), cv::IMREAD_COLOR);
    return !img1.empty() && !img2.empty();
}

// ---------- CSV logger ----------
static void append_csv(const fs::path & csv, const MetricsRow & r, const std::string & setName) {
    bool exists = fs::exists(csv);
    std::ofstream f(csv, std::ios::app);
    if (!exists) f << "set,method,kp1,kp2,matches_raw,matches_final,inliers,match_ms,ransac_thresh,reproj_err,overlap_err_overlay\n";
    f << setName << "," << r.method << "," << r.kp1 << "," << r.kp2 << "," << r.matches_raw << "," << r.matches_final
        << "," << r.inliers << "," << r.match_ms << "," << r.ransac_thresh << "," << r.reproj_err << "," << r.overlap_err_overlay << "\n";
}

int main(int argc, char** argv) {
    try {
        fs::path data_root = (argc > 1 ? fs::path(argv[1]) : fs::path("data"));
        fs::path out_root = (argc > 2 ? fs::path(argv[2]) : fs::path("results"));

        std::vector<std::string> methods = { "AKAZE", "ORB" };
        std::vector<double> ransac_thresh = { 1.5, 3.0, 5.0 };

        // Discover sets automatically: directories named set
        std::vector<fs::path> sets;
        if (fs::exists(data_root)) {
            for (auto& e : fs::directory_iterator(data_root)) {
                if (e.is_directory() && e.path().filename().string().rfind("set", 0) == 0) sets.push_back(e.path());
            }
        }
        if (sets.empty()) {
            std::cerr << "No sets found under " << data_root << " (expected set1, set2, set3 with two images each).\n";
            return 2;
        }
        std::sort(sets.begin(), sets.end());

        fs::path csv = out_root / "results.csv"; if (fs::exists(csv)) fs::remove(csv);

        for (auto& setdir : sets) {
            cv::Mat img1, img2; if (!load_pair_from_set(setdir, img1, img2)) { std::cerr << "Failed to load two images from " << setdir << "\n"; continue; }
            std::string setName = setdir.filename().string();
            std::cout << "Processing " << setName << " (" << img1.cols << "x" << img1.rows << ") + (" << img2.cols << "x" << img2.rows << ")\n";

            for (const auto& method : methods) {
                for (double th : ransac_thresh) {
                    fs::path outdir = out_root / method / setName / ("thr_" + std::to_string((int)std::round(th * 10)));
                    ensure_dir(outdir);
                    auto r = process_pair(outdir, img1, img2, method, th);
                    append_csv(csv, r, setName);
                }
            }
        }

        std::cout << "Done. Results saved under: " << out_root << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl; return 1;
    }
}
