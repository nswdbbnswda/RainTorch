#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_set>
#include <stdexcept>

// ===================== 工具函数 =====================
inline int numel(const std::vector<int>& shape) {
    int cnt = 1;
    for (int d : shape) cnt *= d;
    return cnt;
}

std::vector<int> broadcast_shape(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> res;
    int i = (int)a.size() - 1, j = (int)b.size() - 1;
    while (i >= 0 || j >= 0) {
        int da = (i >= 0) ? a[i] : 1;
        int db = (j >= 0) ? b[j] : 1;
        if (da != 1 && db != 1 && da != db) {
            throw std::runtime_error("broadcast failed: shape mismatch");
        }
        res.push_back(std::max(da, db));
        i--; j--;
    }
    std::reverse(res.begin(), res.end());
    return res;
}

int coord2offset(const std::vector<int>& coord, const std::vector<int>& shape) {
    int off = 0;
    int stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        off += coord[i] * stride;
        stride *= shape[i];
    }
    return off;
}

int broadcast_offset(const std::vector<int>& out_coord,
                     const std::vector<int>& out_shape,
                     const std::vector<int>& in_shape) {
    std::vector<int> in_coord(in_shape.size(), 0);
    int diff = (int)out_shape.size() - (int)in_shape.size();
    for (int i = 0; i < (int)in_shape.size(); ++i) {
        int out_dim_idx = diff + i;
        int out_c = out_coord[out_dim_idx];
        int in_d = in_shape[i];
        in_coord[i] = (in_d == 1) ? 0 : out_c;
    }
    return coord2offset(in_coord, in_shape);
}

// ===================== Node 张量节点 =====================
struct Node {
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<int> shape;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> grad_fn;
    bool requires_grad;
    bool is_leaf;

    Node(std::vector<double> d, std::vector<int> shp, bool req_grad = true, bool leaf = false)
        : data(std::move(d)), shape(std::move(shp)), requires_grad(req_grad), is_leaf(leaf)
    {
        int n = numel(shape);
        grad.assign(n, 0.0);
    }
};
using Var = std::shared_ptr<Node>;

// 创建张量
Var create(std::vector<double> data, std::vector<int> shape, bool requires_grad = true, bool is_leaf = false) {
    return std::make_shared<Node>(std::move(data), std::move(shape), requires_grad, is_leaf);
}
Var create(double val, bool requires_grad = true, bool is_leaf = false) {
    return create({val}, {1}, requires_grad, is_leaf);
}

// ===================== 自动微分拓扑 & 反向传播 =====================
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

void backward(Var out) {
    std::vector<Var> topo;
    std::unordered_set<Node*> visited;
    build_topo(out, topo, visited);
    // 输出梯度置1
    std::fill(out->grad.begin(), out->grad.end(), 1.0);
    // 逆序执行梯度函数
    std::reverse(topo.begin(), topo.end());
    for (auto& n : topo) {
        if (n->grad_fn) n->grad_fn();
    }
    // 清空非叶子梯度
    for (auto& n : topo) {
        if (!n->is_leaf) {
            std::fill(n->grad.begin(), n->grad.end(), 0.0);
        }
    }
}

// ===================== 多维广播四则算子 =====================
Var operator+(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int out_n = numel(out_shape);
    std::vector<double> out_data(out_n, 0.0);
    std::vector<int> coord(out_shape.size(), 0);

    for (int idx = 0; idx < out_n; ++idx) {
        int tmp = idx;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int off_a = broadcast_offset(coord, out_shape, a->shape);
        int off_b = broadcast_offset(coord, out_shape, b->shape);
        out_data[idx] = a->data[off_a] + b->data[off_b];
    }
    Var res = create(out_data, out_shape, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res, out_shape]() {
            int n = numel(out_shape);
            std::vector<int> coord(out_shape.size(), 0);
            for (int idx = 0; idx < n; ++idx) {
                int tmp = idx;
                for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                double dout = res->grad[idx];
                int off_a = broadcast_offset(coord, out_shape, a->shape);
                int off_b = broadcast_offset(coord, out_shape, b->shape);
                a->grad[off_a] += dout;
                b->grad[off_b] += dout;
            }
        };
    }
    return res;
}

Var operator*(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int out_n = numel(out_shape);
    std::vector<double> out_data(out_n, 0.0);
    std::vector<int> coord(out_shape.size(), 0);

    for (int idx = 0; idx < out_n; ++idx) {
        int tmp = idx;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int off_a = broadcast_offset(coord, out_shape, a->shape);
        int off_b = broadcast_offset(coord, out_shape, b->shape);
        out_data[idx] = a->data[off_a] * b->data[off_b];
    }
    Var res = create(out_data, out_shape, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res, out_shape]() {
            int n = numel(out_shape);
            std::vector<int> coord(out_shape.size(), 0);
            for (int idx = 0; idx < n; ++idx) {
                int tmp = idx;
                for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                double dout = res->grad[idx];
                int off_a = broadcast_offset(coord, out_shape, a->shape);
                int off_b = broadcast_offset(coord, out_shape, b->shape);
                a->grad[off_a] += dout * b->data[off_b];
                b->grad[off_b] += dout * a->data[off_a];
            }
        };
    }
    return res;
}

Var operator-(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int out_n = numel(out_shape);
    std::vector<double> out_data(out_n, 0.0);
    std::vector<int> coord(out_shape.size(), 0);

    for (int idx = 0; idx < out_n; ++idx) {
        int tmp = idx;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int off_a = broadcast_offset(coord, out_shape, a->shape);
        int off_b = broadcast_offset(coord, out_shape, b->shape);
        out_data[idx] = a->data[off_a] - b->data[off_b];
    }
    Var res = create(out_data, out_shape, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res, out_shape]() {
            int n = numel(out_shape);
            std::vector<int> coord(out_shape.size(), 0);
            for (int idx = 0; idx < n; ++idx) {
                int tmp = idx;
                for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                double dout = res->grad[idx];
                int off_a = broadcast_offset(coord, out_shape, a->shape);
                int off_b = broadcast_offset(coord, out_shape, b->shape);
                a->grad[off_a] += dout;
                b->grad[off_b] -= dout;
            }
        };
    }
    return res;
}

Var operator/(Var a, Var b) {
    bool need_grad = a->requires_grad || b->requires_grad;
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int out_n = numel(out_shape);
    std::vector<double> out_data(out_n, 0.0);
    std::vector<int> coord(out_shape.size(), 0);

    for (int idx = 0; idx < out_n; ++idx) {
        int tmp = idx;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int off_a = broadcast_offset(coord, out_shape, a->shape);
        int off_b = broadcast_offset(coord, out_shape, b->shape);
        out_data[idx] = a->data[off_a] / b->data[off_b];
    }
    Var res = create(out_data, out_shape, need_grad);
    res->parents = {a, b};
    if (need_grad) {
        res->grad_fn = [a, b, res, out_shape]() {
            int n = numel(out_shape);
            std::vector<int> coord(out_shape.size(), 0);
            for (int idx = 0; idx < n; ++idx) {
                int tmp = idx;
                for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                double dout = res->grad[idx];
                int off_a = broadcast_offset(coord, out_shape, a->shape);
                int off_b = broadcast_offset(coord, out_shape, b->shape);
                double va = a->data[off_a];
                double vb = b->data[off_b];
                a->grad[off_a] += dout / vb;
                b->grad[off_b] -= dout * va / (vb * vb);
            }
        };
    }
    return res;
}

Var operator-(Var x) {
    bool need_grad = x->requires_grad;
    int n = numel(x->shape);
    std::vector<double> out_data(n);
    for (int i = 0; i < n; ++i) out_data[i] = -x->data[i];
    Var res = create(out_data, x->shape, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res]() {
            int n = numel(res->shape);
            for (int i = 0; i < n; ++i) {
                x->grad[i] -= res->grad[i];
            }
        };
    }
    return res;
}

// ReLU
Var relu(Var x) {
    bool need_grad = x->requires_grad;
    int n = numel(x->shape);
    std::vector<double> out_data(n);
    for (int i = 0; i < n; ++i) {
        out_data[i] = x->data[i] > 0 ? x->data[i] : 0.0;
    }
    Var res = create(out_data, x->shape, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res]() {
            int n = numel(res->shape);
            for (int i = 0; i < n; ++i) {
                if (x->data[i] > 0) {
                    x->grad[i] += res->grad[i];
                }
            }
        };
    }
    return res;
}

// 逐元素平方
Var square(Var x) {
    bool need_grad = x->requires_grad;
    int n = numel(x->shape);
    std::vector<double> out_data(n);
    for (int i = 0; i < n; ++i) {
        out_data[i] = x->data[i] * x->data[i];
    }
    Var res = create(out_data, x->shape, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res]() {
            int n = numel(res->shape);
            for (int i = 0; i < n; ++i) {
                x->grad[i] += res->grad[i] * 2 * x->data[i];
            }
        };
    }
    return res;
}

// 求和所有元素
Var sum(Var x) {
    bool need_grad = x->requires_grad;
    int n = numel(x->shape);
    double s = 0.0;
    for (double v : x->data) s += v;
    Var res = create(s, need_grad);
    res->parents = {x};
    if (need_grad) {
        res->grad_fn = [x, res]() {
            double dout = res->grad[0];
            int n = numel(x->shape);
            for (int i = 0; i < n; ++i) {
                x->grad[i] += dout;
            }
        };
    }
    return res;
}

// ===================== SGD 优化器 =====================
struct SGD {
    double lr;
    std::vector<Var> params;
    SGD(double lr, const std::vector<Var>& params) : lr(lr), params(params) {}

    void step() {
        for (auto& v : params) {
            int n = numel(v->shape);
            for (int i = 0; i < n; ++i) {
                v->data[i] -= lr * v->grad[i];
            }
        }
    }
    void zero_grad() {
        for (auto& v : params) {
            if (v->is_leaf) {
                std::fill(v->grad.begin(), v->grad.end(), 0.0);
            }
        }
    }
};

// ===================== Module 网络基类 =====================
struct Module {
    virtual ~Module() = default;
    virtual Var forward(Var x) = 0;
    virtual std::vector<Var> parameters() {
        std::vector<Var> res;
        for (auto& m : subs) {
            auto p = m->parameters();
            res.insert(res.end(), p.begin(), p.end());
        }
        return res;
    }
    void register_submodule(std::shared_ptr<Module> sub) {
        subs.push_back(sub);
    }
private:
    std::vector<std::shared_ptr<Module>> subs;
};

// 单层线性（仅标量版本，演示用；如需真正矩阵乘需补充matmul）
struct Linear : Module {
    Var w;
    Var b;
    Linear(int, int) {
        w = create(0.0, true, true);
        b = create(0.0, true, true);
    }
    Var forward(Var x) override {
        return w * x + b;
    }
    std::vector<Var> parameters() override {
        return {w, b};
    }
};

// 两层网络
struct TwoLayerNet : Module {
    std::shared_ptr<Linear> fc1, fc2;
    TwoLayerNet() {
        fc1 = std::make_shared<Linear>(1, 4);
        fc2 = std::make_shared<Linear>(4, 1);
        register_submodule(fc1);
        register_submodule(fc2);
    }
    Var forward(Var x) override {
        x = fc1->forward(x);
        x = relu(x);
        x = fc2->forward(x);
        return x;
    }
};

// ===================== 训练入口 =====================
int main() {
    // 数据集 y = 3x + 4
    std::vector<double> xs = {1, 2, 3, 4, 5};
    std::vector<double> ys = {7, 10, 13, 16, 19};

    Linear model(1, 1);
    auto params = model.parameters();
    SGD sgd(0.01, params);

    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        sgd.zero_grad();
        Var total_loss = create(0.0, false);

        for (int i = 0; i < (int)xs.size(); ++i) {
            Var x = create(xs[i], false);
            Var yt = create(ys[i], false);
            Var yp = model.forward(x);
            Var loss = square(yp - yt);
            total_loss = total_loss + loss;
        }
        backward(total_loss);
        sgd.step();

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch
                      << " w=" << model.w->data[0]
                      << " b=" << model.b->data[0]
                      << " loss=" << total_loss->data[0] << "\n";
        }
    }
    std::cout << "\nTrain finish:\n";
    std::cout << "w ≈ " << model.w->data[0] << "\n";
    std::cout << "b ≈ " << model.b->data[0] << "\n";
    std::cout << "Target w=3, b=4\n";
    return 0;
}
