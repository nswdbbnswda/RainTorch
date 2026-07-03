#include <iostream>
#include "autograd/node.h"
using namespace raintorch;

int main() {
    auto a = std::make_shared<Node>(2.0, true);
    std::cout << "a data: " << a->data << std::endl;
    return 0;
}
