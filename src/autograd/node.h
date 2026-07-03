#ifndef NODE_H
#define NODE_H

#include <vector>
#include <functional>
#include <string>
#include <memory>  // 必须包含这个头文件

namespace raintorch {

// 加上 std:: 命名空间
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

#endif
