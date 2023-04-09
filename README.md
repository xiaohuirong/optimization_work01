## 论文Energy-Efficient Data Collection for UAV-Assisted IoT: Joint Trajectory and Resource Optimization 复现代码

**目前实现了轨迹优化、调度优化、能量优化部分**

- 系统环境：Archlinux, python3.10

- 文件目录：

  ```txt
  .
  ├── dataload.py # 加载初始数据和生成一些随机数据
  ├── initData.mat # 初始数据
  ├── main.py # 主函数
  ├── myplot.py # 绘图函数
  ├── power.py # 能量优化部分
  ├── README.md 
  ├── schedule.py # 调度优化部分
  └── trajectory.py # 轨迹优化部分
  ```

- 使用方法：

  1. 安装依赖

     ```shell
     pip install numpy matplotlib
     pip install cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS]
     ```
  
  2. 运行
  
     ```shell
     python main.py
     ```
  
     

![轨迹优化](https://raw.github.com/xiaohuirong/images/main/tra_opt.png)

