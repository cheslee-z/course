/*
题目1：实现四元数插值方法NLerp。
描述：一个物体在t1时刻的姿态用四元数q1描述，在t2时刻用q2描述，求在t时刻（t1 <= t <= t2）的姿态四元数q。
*/
#include <iostream>
#include <cmath>

using namespace std;

// 定义四元数结构体
struct Q{
    double x;
    double y;
    double z;
    double w;
};

Q NLerp(const Q& q1, const Q& q2, const double& t)
{
    Q q;
    double q_norm;

    q.x = (1.0 -t)*q1.x+t*q2.x;
    q.y = (1.0 -t)*q1.y+t*q2.y;
    q.z = (1.0 -t)*q1.z+t*q2.z;
    q.w = (1.0 -t)*q1.w+t*q2.w;

    q_norm = sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);

    q.x = q.x / q_norm;
    q.y = q.y / q_norm;
    q.z = q.z / q_norm;
    q.w = q.w / q_norm;

    return q;

}

int main()
{
    Q q1;
    q1.x = 0.5;
    q1.y = 0.5;
    q1.z = 0.5;
    q1.w = 0.5;
    double t1 = 313109;

    Q q2;
    q2.x = 0.5;
    q2.y = -0.5;
    q2.z = -0.5;
    q2.w = 0.5;
    double t2 = 653523;

    double t = 442531;
    double norm_t ;
    norm_t = (t-t1)/(t2-t1);

    Q q;
    q = NLerp(q1,q2,norm_t);
    cout << q.x << endl << q.y << endl << q.z << endl << q.w;
    return 0;
}
