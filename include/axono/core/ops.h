// axono/core/ops.h
#pragma once

#include <pybind11/pybind11.h>
#include <functional>
#include <unordered_map>
#include <string>

namespace axono {
namespace core {

using OpFunction = std::function<pybind11::object(const pybind11::args&)>;

class OpRegistry {
public:
    static OpRegistry& instance() {
        static OpRegistry registry;
        return registry;
    }

    void register_op(const std::string& name, OpFunction func) {
        ops_[name] = std::move(func);
    }

    const OpFunction& get_op(const std::string& name) const {
        auto it = ops_.find(name);
        if (it == ops_.end()) {
            throw std::runtime_error("算子 " + name + " 不存在。");
        }
        return it->second;
    }

    void bind_all(pybind11::module& m) {
        for (const auto& [name, func] : ops_) {
            m.def(name.c_str(), [func](const pybind11::args& args) {
                return func(args);
            });
        }
    }

private:
    OpRegistry() = default;
    std::unordered_map<std::string, OpFunction> ops_;
};

#define REGISTER_OP(name) \
    struct RegisterOp_##name { \
        RegisterOp_##name() { \
            axono::core::OpRegistry::instance().register_op( \
                #name, [](const pybind11::args& args) { \
                    return op_impl_##name(args); \
                }); \
        } \
    }; \
    static RegisterOp_##name register_op_##name; \
    pybind11::object op_impl_##name(const pybind11::args& args)

} // namespace core
} // namespace axono
