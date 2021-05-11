/*
题目4：精密仪器厂，接到订单，需提供总重量为x kg的砝码标准件，厂内仓库中，有足量的1kg，3kg， 5kg，7kg的砝码标准件。需组合成客户所需的x kg。请计算一共有多少种组合方式。
*/
#include <iostream>
#include <vector>

using namespace std;

int Weights[4] = {1, 3, 5, 7};

int main()
{
    int x;
    cout << "x=";
    cin >> x ;

    vector<int> dp(x + 1);
    dp[0] = 1;

    for (int i = 0; i < 4; i++) 
    {
        int weight = Weights[i];
        for (int v = weight; v <= x; v++) 
        {
              dp[v] = dp[v] + dp[v - weight];
        }
    }

    cout << dp[x];
    return 0;
}
