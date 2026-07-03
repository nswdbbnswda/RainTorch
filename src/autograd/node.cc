#include "node.h"
#include <algorithm>

namespace raintorch {

Node::Node(double val, bool need_grad)
    : data(val), grad(0.0), requires_grad(need_grad), backward_fn([](){})
{
}

void Node::backward() {
    // 1. 损失节点梯度初始化为 1
    this->grad = 1.0;

    // 2. DFS 生成拓扑序列
    std::vector<std::shared_ptr<Node>> topo_order;
    std::vector<bool> visited;
    build_topo(shared_from_this(), topo_order, visited);

    // 3. 逆序遍历拓扑，执行每个节点的反向函数
    std::reverse(topo_order.begin(), topo_order.end());
    for (auto& node : topo_order) {
        node->backward_fn();
    }
}

void Node::build_topo(std::shared_ptr<Node> node,
                      std::vector<std::shared_ptr<Node>>& topo,
                      std::vector<bool>& visited) {
    // 简单方式：用地址判断是否访问过，也可以用哈希表
    auto it = std::find(topo.begin(), topo.end(), node);
    if (it != topo.end()) {
        return;
    }

    // 递归遍历所有前驱节点
    for (auto& prev : node->prev_nodes) {
        build_topo(prev, topo, visited);
    }

    topo.push_back(node);
}

} // namespace raintorch
