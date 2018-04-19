#include "cuda.h"

GpuMat GpuMat_New() {
    return new cv::cuda::GpuMat();
}

void GpuMat_Close(GpuMat m) {
    delete m;
    m = nullptr;
}

void GpuMat_Upload(GpuMat gm, Mat m) {
    gm->upload(*m);
}

Mat GpuMat_ConvertTo(GpuMat gm) {
    return (new cv::Mat(*gm));
}

HOG HOG_Create() {
    auto hog = cv::cuda::HOG::create();
    return new cv::Ptr<cv::cuda::HOG>(hog);
}

HOG HOG_Create_WithParams(Size win_size, Size block_size, Size block_stride, Size cell_size, int nbins) {
    cv::Size cv_win_size(win_size.width, win_size.height);
    cv::Size cv_block_size(block_size.width, block_size.height);
    cv::Size cv_block_stride(block_stride.width, block_stride.height);
    cv::Size cv_cell_size(cell_size.width, cell_size.height);

    auto hog = cv::cuda::HOG::create(cv_win_size, cv_block_size, cv_block_stride, cv_cell_size, nbins);
    return new cv::Ptr<cv::cuda::HOG>(hog);
}

void HOG_Close(HOG h) {
    delete h;
    h = nullptr;
}

void HOG_SetSVMDetector(HOG h, Mat detector) {
    (*h)->setSVMDetector(*detector);
}

Rects FromVecRect(std::vector<cv:Rect> vec_rect) {
    Rect* rects = new Rect[vec_rect.size()];
    for (size_t i = 0; i < vec_rect.size(); ++i) {
        Rect r = {vec_rect[i].x, vec_rect[i].y, vec_rect[i].width, vec_rect[i].height};
        rects[i] = r;
    }

    Rects ret = {rects, (int) vec_rect.size()};
    return ret;
}

VecDouble FromVecDouble(std::vector<double> vec_double) {
    double* ds = new double[vec_double.size()];
    for (size_t i = 0; i < vec_double.size(); ++i) {
        ds[i] = vec_double[i];
    }

    Rects ret = {ds, (int) vec_double.size()};
    return ret;
}

Rects HOG_DetectMultiScale(HOG h, GpuMat image) {
    std::vector<cv::Rect> detected;
    (*h)->detectMultiScale(*img, detected);
    return FromVecRect(detected);
}

Rects HOG_DetectMultiScale_WithConf(HOG h, GpuMat image, VecDouble* confidence) {
    (*h)->setGroupThreshold(0);
    std::vector<cv::Rect> vec_rects;
    std::vector<double> vec_confidences;
    (*hog)->setGroupThreshold(0);
    (*hog)->detectMultiScale(*image, vec_rects, vec_confidences);
    confidence = FromVecDouble(vec_confidences);
    return FromVecRect(vec_rects);
}

void HOG_SetGammaCorrection(HOG h, bool gamma) {
    (*h)->setGammaCorrection(gamma);
}

void HOG_SetGroupThreshold(HOG h, int group_threshold) {
    (*h)->setGroupThreshold(group_threshold);
}

void Hog_SetHitThreshold(HOG h, double hit_threshold) {
    (*h)->setHitThreshold(hit_threshold);
}

void HOG_SetL2hysThreshold(HOG h, double l2hys_threshold) {
    (*h)->setL2hysThreshold(l2hys_threshold);
}

void HOG_SetNumLevels(HOG h, int num_levels) {
    (*h)->setNumLevels(num_levels);
}

void HOG_SetScaleFactor(HOG h, double scale_factor) {
    (*h)->setScaleFactor(scale_factor);
}

void HOG_SetWinSigma(HOG h, double win_sigma) {
    (*h)->setWinSigma(win_sigma);
}

void HOG_SetWinStride(HOG h, Size win_stride) {
    (*h)->setWinStride(win_stride);
}

bool HOG_GetGammaCorrection(HOG h) {
    return (*h)->getGammaCorrection();
}

int HOG_GetGroupThreshold(HOG h) {
    return (*h)->getGroupThreshold();
}

double HOG_GetHitThreshold(HOG h) {
    return (*h)->getHitThreshold();
}

double HOG_GetL2hysThreshold(HOG h) {
    return (*h)->getL2hysThreshold();
}

int HOG_GetNumLevels(HOG h) {
    return (*h)->getNumLevels();
}

double HOG_GetScaleFactor(HOG h) {
    return (*h)->getScaleFactor();
}

double HOG_GetWinSigma(HOG h) {
    return (*h)->getWinSigma();
}

Size HOG_GetWinStride(HOG h) {
    return (*h)->getWinStride();
}

GpuCascade GpuCascade_Create(const char* const filename) {
    auto cascade = cv::cuda::CascadeClassifier::create(filename);
    return new cv::Ptr<cv::cuda::CascadeClassifier>(cascade);
}

void GpuCascade_Close(GpuCascade cascade) {
    delete cascade;
    cascade = nullptr;
}

Rects GpuCascade_DetectMultiScale(GpuCascade cascade, GpuMat mat) {
    cv::cuda::GpuMat objbuf;
    std::vector<cv::Rect> vec_object;

    (*cascade)->detectMultiScale(*image, objbuf);
    (*cascade)->convert(objbuf, vec_object);

    return FromVecRect(vec_object);
}

void GpuCascade_SetFindLargestObject(GpuCascade cascade, bool largest) {
    (*cascade)->setFindLargestObject(largest);
}

void GpuCascade_SetMaxNumObjects(GpuCascade cascade, int max) {
    (*cascade)->setMaxNumObjects(max);
}

void GpuCascade_SetMinNeighbors(GpuCascade cascade, int min_neighbors) {
    (*cascade)->setMinNeighbors(min_neighbors);
}

void GpuCascade_SetMaxObjectSize(GpuCascade cascade, Size max_size) {
    cv::Size cv_max_size(max_size.width, max_size.height);
    (*cascade)->setMaxObjectSize(max_size);
}

void GpuCascade_SetMinObjectSize(GpuCascade cascade, Size min_size) {
    cv::Size cv_min_size(min_size.width, min_size.height);
    (*cascade)->setMinObjectSize(max_size);
}

void GpuCascade_SetScaleFactor(GpuCascade cascade, double scale) {
    (*cascade)->setScaleFactor(scale);
}

Size GpuCascade_GetClassifierSize(GpuCascade cascade) {
    cv::Size size = (*cascade)->getClassifierSize();
    return Size { size.width, size.height };
}

bool GpuCascade_GetFindLargestObject(GpuCascade cascade) {
    return (*cascade)->getFindLargestObject();
}

int GpuCascade_GetMaxNumObjects(GpuCascade cascade) {
    return (*cascade)->getMaxNumObjects();
}
int GpuCascade_GetMinNeighbors(GpuCascade cascade) {
    return (*cascade)->getMinNeighbors();
}

Size GpuCascade_GetMaxObjectSize(GpuCascade cascade) {
    cv::Size size = (*cascade)->getMaxObjectSize();
    return Size { size.width, size.height };
}

Size GpuCascade_GetMinObjectSize(GpuCascade cascade) {
    cv::Size size = (*cascade)->getMinObjectSize();
    return Size { size.width, size.height };
}

double GpuCascade_GetScaleFactor(GpuCascade cascade) {
    return (*cascade)->getScaleFactor();
}

#ifdef __cplusplus
}
#endif
