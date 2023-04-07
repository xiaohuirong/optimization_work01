## 论文Energy-Efficient Data Collection for UAV-Assisted IoT: Joint Trajectory and Resource Optimization 复现代码

**目前只实现了轨迹优化部分**

- 系统环境：Archlinux, python3.10

- 文件目录：

  ```txt
  ├── dataload.py # 加载数据
  ├── initData.mat # 初始数据
  ├── main.py # 主函数
  ├── myplot.py # 绘图函数
  ├── README.md 
  ├── schedule.py # 调度优化(待调整)
  └── trajectory.py # 轨迹优化
  
  2 directories, 11 files
  ```

- 使用方法：

  1. 安装依赖

     ```shell
     pip install cvxpy numpy matplotlib
     ```

  2. 运行

     ```shell
     python main.py
     ```

     

![轨迹优化](https://raw.github.com/xiaohuirong/images/main/tra_opt.png)

