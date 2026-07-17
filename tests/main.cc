#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_set>


struct Node {
    double val;
    double grad;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> grad_fn;
    bool requires_grad;

    Node(double v, bool requires_grad=true) : val(v), grad(0.0), requires_grad(requires_grad) {}
};

using Var = std::shared_ptr<Node>;


Var create(double val, bool requires_grad=true) {
    return std::make_shared<Node>(val, requires_grad);
}

struct SGD {
  double lr;
  std::vector<Var> params;
  SGD(double lr, const std::vector<Var>& params) : lr(lr), params(params) {}
  void step() {
    for (auto& v : params) {
      v->val = v->val - lr * v->grad;
    }
  }

  void zero_grad() {
    for (auto& v : params) {
      v->grad = 0;
    }
  }
};

void build_topo(Var node, std::vector<Var>& topo, std::unordered_set<Node*>& visited) {
    if (!node) return;
    Node* raw = node.get();
    if (visited.count(raw)) return;
    visited.insert(raw);

    for (auto& p : node->parents) {
        build_topo(p, topo, visited);
    }
    if (node->requires_grad) {
      topo.push_back(node);
    }
}

void backward(Var out, std::vector<Var>& topo) {
    if (!out) return;
    out->grad = 1.0;

    std::unordered_set<Node*> visited;
    build_topo(out, topo, visited);

    std::reverse(topo.begin(), topo.end());
    for (auto& n : topo) {
      if (n->grad_fn) {
        n->grad_fn();
      }
    }
}

// 四则算子
Var operator+(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    Var res = create(a->val + b->val, need_grad);
    res->parents = {a, b};
    if (need_grad) {
      res->grad_fn = [a, b, res]() {
          a->grad += res->grad;
          b->grad += res->grad;
      };
    }
    return res;
}

Var operator*(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    Var res = create(a->val * b->val, need_grad);
    res->parents = {a, b};
    if (need_grad) {
      res->grad_fn = [a, b, res]() {
        a->grad += res->grad * b->val;
        b->grad += res->grad * a->val;
      };
    }
    return res;
}

Var operator-(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad; 
    Var res = create(a->val - b->val, need_grad);
    res->parents = {a, b};
    if (need_grad) {
      res->grad_fn = [a, b, res](){
          a->grad += res->grad;
          b->grad -= res->grad;
      };
    }
    return res;
}

// 平方，用于损失函数
Var square(Var x) {
    bool need_grad = x->requires_grad;
    Var res = create(x->val * x->val, need_grad);
    res->parents = {x};
    if (need_grad) {
      res->grad_fn = [x, res](){
        x->grad += res->grad * 2 * x->val;
      };
    }
    return res;
}

int main() {
    // 构造数据集 y = 3*x + 4
    std::vector<double> xs = {1, 2, 3, 4, 5};
    std::vector<double> ys = {7, 10, 13, 16, 19};

    // 可训练参数
    Var w = create(0.0);
    Var b = create(0.0);
    double lr = 0.01;
    std::vector<Var> params{w, b};
    SGD sgd(lr, params);

    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        sgd.zero_grad();
        Var total_loss = create(0.0, false);
        std::vector<Var> topo_nodes;
        for (int i = 0; i < xs.size(); i++) {
            double xi = xs[i];
            double yi_true = ys[i];

            Var x = create(xi, false);
            Var y_true = create(yi_true, false);

            Var y_pred = w * x + b;
            Var loss = square(y_pred - y_true);

            total_loss = total_loss + loss;
        }

        backward(total_loss, topo_nodes);
        // SGD 更新
        sgd.step();

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch
                      << " | w=" << w->val
                      << " | b=" << b->val
                      << " | total_loss=" << total_loss->val << "\n";
        }
    }

    std::cout << "\n训练完成！\n";
    std::cout << "拟合得到 w ≈ " << w->val << "\n";
    std::cout << "拟合得到 b ≈ " << b->val << "\n";
    std::cout << "理论真值 w=3, b=4\n";
    return 0;
}
