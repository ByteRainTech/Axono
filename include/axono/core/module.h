#include "tensor.h"

namespace axono::core {
class Module {
private:
    std::unordered_map<std::string, Tensor> weights_;  // 存储权重张量
public:
    void add_weight(const std::string& name, const Tensor& weight) {
        weights_[name] = weight;
    }
    Tensor& get_weight(const std::string& name) {
        return weights_.at(name);
    }
    auto& weights() { return weights_; }
};
}
