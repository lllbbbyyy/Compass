#include "rapidcsv.h"
#include <iostream>
using namespace std;

int main()
{

    // 第二步：用 rapidcsv 打开并写入
    rapidcsv::Document doc;
	doc.SetColumnName(0,"latency");
	doc.SetColumnName(1,"energy");
    doc.SetColumn("latency", std::vector<int>{1, 2, 3});
    doc.SetColumn("energy",  std::vector<int>{4, 5, 6});
    doc.Save("tmp/123.csv");
}