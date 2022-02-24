#include <Python.h>
#include <torch/script.h>

#include "knn_cpu.h"
#include "knn_cuda.h"


torch::Tensor knn(torch::Tensor x, torch::Tensor y,
                  torch::optional<torch::Tensor> ptr_x,
                  torch::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
                  int64_t num_workers) {
  if (x.device().is_cuda()) {
    return knn_cuda(x, y, ptr_x, ptr_y, k, cosine);
  } else {
    if (cosine)
      AT_ERROR("`cosine` argument not supported on CPU");
    return knn_cpu(x, y, ptr_x, ptr_y, k, num_workers);
  }
}

static auto registry =
    torch::RegisterOperators().op("my_torch_cluster::knn", &knn);
