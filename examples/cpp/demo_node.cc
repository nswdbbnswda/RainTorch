#include <iostream>
#include "autograd/node.h"
#include "autograd/ops.h"

using namespace raintorch;

int main() {
    // 1. 创建两个需要计算梯度的变量
    auto a = std::make_shared<Node>(2.0, true);
    auto b = std::make_shared<Node>(3.0, true);

    std::cout << "初始值 a = " << a->data << ", b = " << b->data << "\n\n";

    // 2. 正向计算 c = a * b
    auto c = a * b;
    std::cout << "c = a * b = " << c->data << "\n";

    // 3. 反向传播求梯度 dc/da、dc/db
    c->backward();
    std::cout << "dc/da = " << a->grad << "\n";
    std::cout << "dc/db = " << b->grad << "\n\n";

    // 4. 梯度必须手动清零，否则会累加
    a->grad = 0.0;
    b->grad = 0.0;

    // 5. 复合表达式 d = a * b + a
    auto d = a * b + a;
    std::cout << "d = a * b + a = " << d->data << "\n";
    d->backward();
    std::cout << "dd/da = " << a->grad << "\n";
    std::cout << "dd/db = " << b->grad << "\n\n";

    // 6. 测试减法、平方算子（MSE损失常用）
    a->grad = 0.0;
    b->grad = 0.0;
    auto diff = a - b;
    auto square_diff = pow2(diff);
    std::cout << "(a - b)^2 = " << square_diff->data << "\n";
    square_diff->backward();
    std::cout << "d[(a-b)²]/da = " << a->grad << "\n";
    std::cout << "d[(a-b)²]/db = " << b->grad << "\n";

    return 0;
}
