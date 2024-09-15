#pragma once
#include <iostream>
#include <array>
class AdjacencyList
{
private:
    static const int MAX_SIZE = 16; // Maximum number of vertices

    struct CountNode {
        int darkCount = 0;
        int brightCount = 0;
        int similarCount = 0; // For performance optimization

    };

    std::array<CountNode, MAX_SIZE> nodes; // Fixed-size array for nodes

public:


};

