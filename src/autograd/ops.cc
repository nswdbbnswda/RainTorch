#include "autograd/ops.h"

namespace raintorch {

NodePtr operator+(const NodePtr& a, const NodePtr& b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    auto out = std::make_shared<Node>(a->data + b->data, need_grad);
    out->prev_nodes = {a, b};
    out->op_name = "add";

    out->backward_fn = [out, a, b]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
}

NodePtr operator*(const NodePtr& a, const NodePtr& b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    auto out = std::make_shared<Node>(a->data * b->data, need_grad);
    out->prev_nodes = {a, b};
    out->op_name = "mul";

    out->backward_fn = [out, a, b]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    return out;
}

// 减法 a - b
NodePtr operator-(const NodePtr& a, const NodePtr& b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    auto out = std::make_shared<Node>(a->data - b->data, need_grad);
    out->prev_nodes = {a, b};
    out->op_name = "sub";

    out->backward_fn = [out, a, b]() {
        a->grad += out->grad;
        b->grad -= out->grad;
    };
    return out;
}

// 取负 -a
NodePtr operator-(const NodePtr& a) {
    bool need_grad = a->requires_grad;
    auto out = std::make_shared<Node>(-a->data, need_grad);
    out->prev_nodes = {a};
    out->op_name = "neg";

    out->backward_fn = [out, a]() {
        a->grad -= out->grad;
    };
    return out;
}

// 平方 a^2，用来写MSE损失
NodePtr pow2(const NodePtr& a) {
    bool need_grad = a->requires_grad;
    auto out = std::make_shared<Node>(a->data * a->data, need_grad);
    out->prev_nodes = {a};
    out->op_name = "pow2";

    out->backward_fn = [out, a]() {
        a->grad += 2 * a->data * out->grad;
    };
    return out;
}

} // namespace raintorch
