#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace cv;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// ----------------------------- Config -----------------------------
struct Config {
    string dir;                // images directory
    string outdir = "output";  // output root
    string detector = "orb";   // orb | akaze
    string blend = "feather";  // overlay | feather
    double ransac_thresh = 3.0;
    bool ratio_test = true;    // Lowe's ratio test
    double ratio = 0.75;       // ratio value
    bool cross_check = false;  // BFMatcher crossCheck
    int hist_bins = 30;
    bool draw_debug = true;    // save debug images
};

// ----------------------------- Utils -----------------------------
static inline bool hasImageExt(const fs::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

vector<Mat> loadImagesSorted(const string& dir) {
    vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        if (hasImageExt(e.path())) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    vector<Mat> imgs;
    for (auto& p : files) {
        Mat im = imread(p.string(), IMREAD_COLOR);
        if (im.empty()) {
            cerr << "Failed to read image: " << p << endl;
            continue;
        }
        imgs.push_back(im);
    }
    return imgs;
}

void ensureDir(const string& d) {
    fs::create_directories(d);
}

string timeNowStr() {
    auto t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return string(buf);
}

void saveCSVHeader(const string& csv) {
    if (fs::exists(csv)) return;
    std::ofstream ofs(csv);
    ofs << "set,step,detector,kp1,kp2,matches,good_matches,inliers,ransac_thresh,match_time_ms,homog_time_ms,warp_time_ms,blend,blend_time_ms\n";
}

void appendCSV(const string& csv, const string& setname, const string& step,
    const string& detector, int kp1, int kp2, int matches, int good_matches,
    int inliers, double ransac_thresh, double match_ms, double H_ms,
    double warp_ms, const string& blend, double blend_ms) {
    std::ofstream ofs(csv, std::ios::app);
    ofs << setname << "," << step << "," << detector << ","
        << kp1 << "," << kp2 << "," << matches << "," << good_matches << ","
        << inliers << "," << ransac_thresh << ","
        << match_ms << "," << H_ms << "," << warp_ms << ","
        << blend << "," << blend_ms << "\n";
}

// Draw a simple histogram image for match distances
void saveHistogram(const vector<float>& vals, int bins, const string& outpath, const string& title = "Distance Histogram") {
    if (vals.empty()) return;
    float vmin = *std::min_element(vals.begin(), vals.end());
    float vmax = *std::max_element(vals.begin(), vals.end());
    if (vmax <= vmin) vmax = vmin + 1.f;

    vector<int> hist(bins, 0);
    for (auto v : vals) {
        int bi = (int)((v - vmin) / (vmax - vmin) * bins);
        if (bi == bins) bi = bins - 1;
        hist[bi]++;
    }
    int W = 800, H = 400, margin = 40;
    Mat canvas(H, W, CV_8UC3, Scalar(255, 255, 255));

    int maxCount = *std::max_element(hist.begin(), hist.end());
    double xstep = (W - 2 * margin) / double(bins);
    for (int i = 0; i < bins; ++i) {
        int h = (int)((H - 2 * margin) * (hist[i] / (double)maxCount));
        rectangle(canvas,
            Point((int)(margin + i * xstep), H - margin - h),
            Point((int)(margin + (i + 1) * xstep - 2), H - margin),
            Scalar(80, 120, 220), FILLED);
    }
    // axes
    line(canvas, Point(margin, H - margin), Point(W - margin, H - margin), Scalar(0, 0, 0), 2);
    line(canvas, Point(margin, H - margin), Point(margin, margin), Scalar(0, 0, 0), 2);

    putText(canvas, title, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    char rng[128]; std::snprintf(rng, sizeof(rng), "min=%.2f max=%.2f", vmin, vmax);
    putText(canvas, rng, Point(20, 55), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(60, 60, 60), 1);

    imwrite(outpath, canvas);
}

// ---------------------- Features & Matching -----------------------
Ptr<Feature2D> makeDetector(const string& name) {
    string n = name;
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    if (n == "orb") {
        return ORB::create(3000); // enough for panoramas
    }
    else if (n == "akaze") {
        return AKAZE::create();   // binary MLDB descriptor by default
    }
    else {
        cerr << "Unknown detector: " << name << " (fallback to ORB)\n";
        return ORB::create(3000);
    }
}

int normTypeFor(const string& detector) {

    return NORM_HAMMING;
}

struct MatchResult {
    vector<KeyPoint> k1, k2;
    Mat d1, d2;
    vector<DMatch> matches_all;
    vector<DMatch> good;      // after ratio/cross-check
    vector<float> distances;  // all distances for histogram
};

MatchResult detectAndMatch(const Mat& im1, const Mat& im2, const Config& cfg, double& match_ms, const string& debugDir, const string& prefix) {
    TickMeter tm; tm.start();
    auto detector = makeDetector(cfg.detector);
    MatchResult R;

    detector->detectAndCompute(im1, noArray(), R.k1, R.d1);
    detector->detectAndCompute(im2, noArray(), R.k2, R.d2);

    tm.stop();
    double feat_ms = tm.getTimeMilli();

    int normType = normTypeFor(cfg.detector);

    tm.reset(); tm.start();
    if (cfg.ratio_test) {
        // KNN + ratio
        BFMatcher matcher(normType, false);
        vector<vector<DMatch>> knn;
        matcher.knnMatch(R.d1, R.d2, knn, 2);
        for (auto& v : knn) {
            if (v.size() >= 2) {
                if (v[0].distance < cfg.ratio * v[1].distance) {
                    R.good.push_back(v[0]);
                }
                R.matches_all.push_back(v[0]); // for dist histogram
                R.distances.push_back(v[0].distance);
            }
            else if (!v.empty()) {
                R.matches_all.push_back(v[0]);
                R.distances.push_back(v[0].distance);
            }
        }
    }
    else if (cfg.cross_check) {
        BFMatcher matcher(normType, true);
        matcher.match(R.d1, R.d2, R.good);
        R.matches_all = R.good;
        for (const auto& m : R.matches_all) R.distances.push_back(m.distance);
    }
    else {
        BFMatcher matcher(normType, false);
        matcher.match(R.d1, R.d2, R.matches_all);
        // take top X% as "good"
        std::sort(R.matches_all.begin(), R.matches_all.end(), [](const DMatch& a, const DMatch& b) {return a.distance < b.distance;});
        int keep = (int)std::ceil(R.matches_all.size() * 0.5);
        keep = std::max(keep, std::min(1000, (int)R.matches_all.size()));
        R.good.assign(R.matches_all.begin(), R.matches_all.begin() + keep);
        for (const auto& m : R.matches_all) R.distances.push_back(m.distance);
    }
    tm.stop();
    match_ms = tm.getTimeMilli();

    if (cfg.draw_debug) {
        ensureDir(debugDir);
        // keypoints
        Mat im1kp, im2kp;
        drawKeypoints(im1, R.k1, im1kp, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(im2, R.k2, im2kp, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        imwrite(debugDir + "/" + prefix + "_im1_keypoints.jpg", im1kp);
        imwrite(debugDir + "/" + prefix + "_im2_keypoints.jpg", im2kp);

        // matches
        Mat m1, m2;
        if (!R.matches_all.empty()) {
            drawMatches(im1, R.k1, im2, R.k2, R.matches_all, m1, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            imwrite(debugDir + "/" + prefix + "_matches_all.jpg", m1);
        }
        if (!R.good.empty()) {
            drawMatches(im1, R.k1, im2, R.k2, R.good, m2, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            imwrite(debugDir + "/" + prefix + "_matches_good.jpg", m2);
        }
        // histogram
        saveHistogram(R.distances, cfg.hist_bins, debugDir + "/" + prefix + "_dist_hist.jpg",
            "Match distance (" + cfg.detector + ")");
    }
    cout << "[Detect] " << cfg.detector << " kp1=" << R.k1.size() << " kp2=" << R.k2.size()
        << " matches=" << R.matches_all.size() << " good=" << R.good.size()
        << " feat_ms=" << feat_ms << " match_ms=" << match_ms << endl;

    return R;
}

// ---------------------- Homography & Warping ----------------------
struct HomogResult {
    Mat H;
    vector<char> inlierMask;
    int inliers = 0;
    double H_ms = 0.0;
};

HomogResult estimateH(const MatchResult& M, double ransac_thresh, const Mat& im1, const Mat& im2, const Config& cfg, const string& debugDir, const string& prefix) {
    vector<Point2f> p1, p2;
    p1.reserve(M.good.size());
    p2.reserve(M.good.size());
    for (auto& m : M.good) {
        p1.push_back(M.k1[m.queryIdx].pt);
        p2.push_back(M.k2[m.trainIdx].pt);
    }

    HomogResult R;
    TickMeter tm; tm.start();
    if (p1.size() >= 4) {
        R.H = findHomography(p2, p1, RANSAC, ransac_thresh, R.inlierMask);
        if (!R.inlierMask.empty())
            R.inliers = std::accumulate(R.inlierMask.begin(), R.inlierMask.end(), 0);
    }
    tm.stop();
    R.H_ms = tm.getTimeMilli();

    if (cfg.draw_debug && !R.H.empty()) {
        // draw inlier matches
        vector<DMatch> inlierMatches;
        for (size_t i = 0; i < M.good.size(); ++i) {
            if (i < R.inlierMask.size() && R.inlierMask[i]) inlierMatches.push_back(M.good[i]);
        }
        Mat vis;
        drawMatches(im1, M.k1, im2, M.k2, inlierMatches, vis, Scalar(60, 200, 60), Scalar::all(-1));
        imwrite(debugDir + "/" + prefix + "_inlier_matches.jpg", vis);
    }

    cout << "[Homog] inliers=" << R.inliers << " / " << M.good.size()
        << " rth=" << ransac_thresh << " H_ms=" << R.H_ms << endl;
    return R;
}

// compute panorama bounding box of warping im2 to im1 coords (with H)
void computeCanvas(const Mat& im1, const Mat& im2, const Mat& H, Rect2f& roi, Mat& Htrans) {
    vector<Point2f> c2 = { {0,0}, {(float)im2.cols,0}, {(float)im2.cols,(float)im2.rows}, {0,(float)im2.rows} };
    vector<Point2f> c2w;
    perspectiveTransform(c2, c2w, H);

    vector<Point2f> all = c2w;
    all.push_back(Point2f(0, 0));
    all.push_back(Point2f((float)im1.cols, 0));
    all.push_back(Point2f((float)im1.cols, (float)im1.rows));
    all.push_back(Point2f(0, (float)im1.rows));

    float minx = std::numeric_limits<float>::max(), miny = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::lowest(), maxy = std::numeric_limits<float>::lowest();
    for (auto& p : all) {
        minx = std::min(minx, p.x); miny = std::min(miny, p.y);
        maxx = std::max(maxx, p.x); maxy = std::max(maxy, p.y);
    }
    // shift so top-left >= (0,0)
    float shiftx = (minx < 0) ? -minx : 0.f;
    float shifty = (miny < 0) ? -miny : 0.f;
    Htrans = (Mat_<double>(3, 3) << 1, 0, shiftx, 0, 1, shifty, 0, 0, 1);
    roi = Rect2f(0, 0, std::ceil(maxx + shiftx), std::ceil(maxy + shifty));
}

// overlay blending
Mat blendOverlay(const Mat& base, const Mat& warp) {
    CV_Assert(base.type() == CV_8UC3 && warp.type() == CV_8UC3);
    Mat out = base.clone();
    for (int y = 0; y < warp.rows; ++y) {
        const Vec3b* wptr = warp.ptr<Vec3b>(y);
        Vec3b* optr = out.ptr<Vec3b>(y);
        for (int x = 0; x < warp.cols; ++x) {
            if (wptr[x] != Vec3b(0, 0, 0)) optr[x] = wptr[x]; // non-black overwrite
        }
    }
    return out;
}

// feather blending using distance transform
Mat blendFeather(const Mat& im1w, const Mat& im2w) {
    CV_Assert(im1w.type() == CV_8UC3 && im2w.type() == CV_8UC3);

    Mat g1, g2; cvtColor(im1w, g1, COLOR_BGR2GRAY); cvtColor(im2w, g2, COLOR_BGR2GRAY);
    Mat m1 = g1 > 0, m2 = g2 > 0; // foreground masks

    Mat dt1, dt2;
    distanceTransform(m1, dt1, DIST_L2, 3);
    distanceTransform(m2, dt2, DIST_L2, 3);

    // weights
    Mat w1, w2;
    dt1.convertTo(w1, CV_32F);
    dt2.convertTo(w2, CV_32F);

    Mat denom;
    denom = w1 + w2 + 1e-6f;
    divide(w1, denom, w1);
    divide(w2, denom, w2);

    Mat f1, f2;
    im1w.convertTo(f1, CV_32FC3);
    im2w.convertTo(f2, CV_32FC3);

    // apply weights per-channel
    vector<Mat> ch1, ch2; split(f1, ch1); split(f2, ch2);
    for (int c = 0;c < 3;++c) {
        ch1[c] = ch1[c].mul(w1);
        ch2[c] = ch2[c].mul(w2);
    }
    Mat sum1, sum2, out32;
    merge(ch1, sum1); merge(ch2, sum2);
    out32 = sum1 + sum2;

    // where only one image exists, ensure value kept even if weight is ~0 due to numerical
    // Already handled since other is 0.

    Mat out8; out32.convertTo(out8, CV_8UC3);
    return out8;
}

struct StitchResult {
    Mat pano;
    double warp_ms = 0, blend_ms = 0;
};

StitchResult stitchPair(const Mat& im1, const Mat& im2, const Mat& H, const Config& cfg) {
    StitchResult SR;
    if (H.empty()) {
        cerr << "Homography is empty; returning im1 as panorama.\n";
        SR.pano = im1.clone();
        return SR;
    }
    Rect2f roi; Mat Ht;
    computeCanvas(im1, im2, H, roi, Ht);

    TickMeter tm; tm.start();
    Mat out(roi.size(), CV_8UC3, Scalar(0, 0, 0));

    Mat im1w, im2w;
    warpPerspective(im2, im2w, Ht * H, roi.size()); // warp im2 into canvas
    warpPerspective(im1, im1w, Ht, roi.size());   // shift im1 into canvas
    tm.stop(); SR.warp_ms = tm.getTimeMilli();

    tm.reset(); tm.start();
    Mat pano;
    if (cfg.blend == "overlay") pano = blendOverlay(im1w, im2w);
    else pano = blendFeather(im1w, im2w);
    tm.stop(); SR.blend_ms = tm.getTimeMilli();

    SR.pano = pano;
    return SR;
}

// ------------------------------ CLI ------------------------------
void printHelp() {
    cout <<
        R"(Usage:
  panorama --dir <image_folder> [--out <output_folder>] [--detector orb|akaze]
           [--blend overlay|feather] [--ransac <thresh>] [--ratio <v>]
           [--no-ratio] [--cross-check]
)";
}

Config parseArgs(int argc, char** argv) {
    Config cfg;
    if (argc < 3) { printHelp(); exit(0); }
    for (int i = 1;i < argc;++i) {
        string a = argv[i];
        if (a == "--dir" && i + 1 < argc) cfg.dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) cfg.outdir = argv[++i];
        else if (a == "--detector" && i + 1 < argc) cfg.detector = argv[++i];
        else if (a == "--blend" && i + 1 < argc) cfg.blend = argv[++i];
        else if (a == "--ransac" && i + 1 < argc) cfg.ransac_thresh = std::stod(argv[++i]);
        else if (a == "--ratio" && i + 1 < argc) { cfg.ratio = std::stod(argv[++i]); cfg.ratio_test = true; }
        else if (a == "--no-ratio") cfg.ratio_test = false;
        else if (a == "--cross-check") cfg.cross_check = true;
        else if (a == "--help") { printHelp(); exit(0); }
        else {
            // ignore unknown
        }
    }
    if (cfg.dir.empty()) { printHelp(); exit(1); }
    std::transform(cfg.detector.begin(), cfg.detector.end(), cfg.detector.begin(), ::tolower);
    std::transform(cfg.blend.begin(), cfg.blend.end(), cfg.blend.begin(), ::tolower);
    return cfg;
}

 //----------------------------- Main ------------------------------
int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);

    string setname = fs::path(cfg.dir).filename().string();
    string runTag = setname + "_" + timeNowStr();
    string outSetDir = cfg.outdir + "/" + runTag;
    string dbgDir = outSetDir + "/debug";
    ensureDir(outSetDir);
    ensureDir(dbgDir);

    string csv = cfg.outdir + "/results.csv";
    saveCSVHeader(csv);

    auto imgs = loadImagesSorted(cfg.dir);
    if (imgs.size() < 2) {
        cerr << "Need at least 2 images in: " << cfg.dir << endl;
        return 1;
    }
    cout << "Loaded " << imgs.size() << " images from " << cfg.dir << endl;

    Mat pano = imgs[0].clone();
    imwrite(outSetDir + "/00_input_0.jpg", pano);

    for (size_t i = 1; i < imgs.size(); ++i) {
        Mat im1 = pano;
        Mat im2 = imgs[i];

        // 1) Features & Matching
        double match_ms = 0.0;
        string stepPrefix = (i < 10 ? ("0" + std::to_string(i)) : std::to_string(i));
        auto MR = detectAndMatch(im1, im2, cfg, match_ms, dbgDir, stepPrefix);

        // 2) Homography with RANSAC
        auto HR = estimateH(MR, cfg.ransac_thresh, im1, im2, cfg, dbgDir, stepPrefix);

        // 3) Stitch & Blend
        auto SR = stitchPair(im1, im2, HR.H, cfg);
        pano = SR.pano;

        // Save intermediates
        imwrite(outSetDir + "/" + stepPrefix + "_warp_im1.jpg", SR.pano); // final after this step
        imwrite(outSetDir + "/" + stepPrefix + "_pano.jpg", pano);

        // record CSV
        appendCSV(csv, setname, "step" + std::to_string(i), cfg.detector,
            (int)MR.k1.size(), (int)MR.k2.size(),
            (int)MR.matches_all.size(), (int)MR.good.size(),
            HR.inliers, cfg.ransac_thresh, match_ms, HR.H_ms,
            SR.warp_ms, cfg.blend, SR.blend_ms);

        cout << "[Step " << i << "] "
            << "inliers=" << HR.inliers
            << " pano_size=" << pano.cols << "x" << pano.rows
            << " warp_ms=" << SR.warp_ms << " blend_ms=" << SR.blend_ms
            << endl;
    }

    // Final save
    imwrite(outSetDir + "/final_panorama.jpg", pano);
    cout << "Done. Outputs at: " << outSetDir << "\nCSV: " << csv << endl;
    return 0;
}

