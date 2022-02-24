#include <Python.h>
#include <torch/script.h>

#include "radius_cpu.h"
#include "radius_cuda.h"


torch::Tensor radius(torch::Tensor x, torch::Tensor y,
                     torch::optional<torch::Tensor> ptr_x,
                     torch::optional<torch::Tensor> ptr_y, double r,
                     int64_t max_num_neighbors, int64_t num_workers) {
  if (x.device().is_cuda()) {
    return radius_cuda(x, y, ptr_x, ptr_y, r, max_num_neighbors);
  } else {
    return radius_cpu(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers);
  }
}

static auto registry =
    torch::RegisterOperators().op("my_torch_cluster::radius", &radius);
