#pragma once

#include <vector>
#include <functional>
#include <string>
#include <memory>

namespace raintorch {

struct Node : std::enable_shared_from_this<Node> {
    double data;
    double grad;
    bool requires_grad;

    std::vector<std::shared_ptr<Node>> prev_nodes;
    std::string op_name;
    std::function<void()> backward_fn;

    explicit Node(double val, bool need_grad = false);
    void backward();

private:
    void build_topo(std::shared_ptr<Node> node,
                    std::vector<std::shared_ptr<Node>>& topo,
                    std::vector<bool>& visited);
};

using NodePtr = std::shared_ptr<Node>;

} // namespace raintorch
