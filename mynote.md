 ```bash
 python setup.py install --user
 ```
 报错：
 ```bash
/home/hefeiwang/miniconda3/envs/ravens/lib/python3.7/site-packages/setuptools/command/install.py:37: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
  setuptools.SetuptoolsDeprecationWarning,
/home/hefeiwang/miniconda3/envs/ravens/lib/python3.7/site-packages/setuptools/command/easy_install.py:147: EasyInstallDeprecationWarning: easy_install command is deprecated. Use build and pip and other standards-based tools.
  EasyInstallDeprecationWarning,
zip_safe flag not set; analyzing archive contents...
```
CSDN搜索发现是因为setuptools版本过高：
```bash
(ravens) hefeiwang@WangHF:~/ravens/ravens$ pip list |grep se
astunparse              1.6.3
charset-normalizer      3.4.0
setuptools              65.6.3
tensorboard-data-server 0.6.1
```
降级为56：
```bash
(ravens) hefeiwang@WangHF:~/ravens/ravens$ pip install setuptools==56
```
问题解决。

```bash
(ravens) hefeiwang@WangHF:~/ravens/ravens$ python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=train --n=10
2024-12-15 15:11:59.597475: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2024-12-15 15:11:59.597530: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
pybullet build time: Nov 28 2023 23:50:19
RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd
Traceback (most recent call last):
  File "ravens/demos.py", line 25, in <module>
    from ravens import tasks
  File "/home/hefeiwang/.local/lib/python3.7/site-packages/ravens-0.1-py3.7.egg/ravens/__init__.py", line 18, in <module>
    from ravens import agents
  File "/home/hefeiwang/.local/lib/python3.7/site-packages/ravens-0.1-py3.7.egg/ravens/agents/__init__.py", line 18, in <module>
    from ravens import agents
  File "/home/hefeiwang/.local/lib/python3.7/site-packages/ravens-0.1-py3.7.egg/ravens/agents/conv_mlp.py", line 22, in <module>
  File "/home/hefeiwang/.local/lib/python3.7/site-packages/ravens-0.1-py3.7.egg/ravens/models/__init__.py", line 18, in <module>
    from ravens import agents
  File "/home/hefeiwang/.local/lib/python3.7/site-packages/ravens-0.1-py3.7.egg/ravens/models/attention.py", line 21, in <module>
  File "/home/hefeiwang/.local/lib/python3.7/site-packages/ravens-0.1-py3.7.egg/ravens/utils/utils.py", line 28, in <module>
ImportError: numpy.core.multiarray failed to import
```
CSDN搜索发现是numpy版本不兼容导致的，查看requirements.txt发现只有pybullet的版本具有一定灵活性，于是查看pybullet版本，果然，自动安装了最新版本Version: 3.2.6，卸载重装requirements.txt中允许的最低版本3.0.4，问题解决。

```bash
(ravens) hefeiwang@WangHF:~/ravens/ravens$ python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=train --n=10
2024-12-15 15:34:51.215738: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
pybullet build time: Sep 22 2020 00:55:20
startThreads creating 1 threads.
starting thread 0
started thread 0
argc=2
argv[0] = --unused
argv[1] = --start_demo_name=Physics Server
ExampleBrowserThreadFunc started
X11 functions dynamically loaded using dlopen/dlsym OK!
X11 functions dynamically loaded using dlopen/dlsym OK!
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
Creating context
Failed to create GL 3.3 context ... using old-style GLX context
Failed to create an OpenGL context
```
找不到swrast_dri.so驱动，参考[CSDN帖子](https://blog.csdn.net/weixin_42092516/article/details/129879122?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-129879122-blog-124125723.235%5Ev38%5Epc_relevant_anti_t3_base&spm=1001.2101.3001.4242.1&utm_relevant_index=1&ydreferer=aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl82MjIxNjg2Mi9hcnRpY2xlL2RldGFpbHMvMTI0MTI1NzIz)找到对应目录，建立软链接
```bash
(ravens) hefeiwang@WangHF:/usr/lib/dri$ sudo ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so swrast_dri.so
```
上一个问题解决，出现新问题：
```bash
(ravens) hefeiwang@WangHF:~/ravens/ravens$ python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=train --n=10
2024-12-15 15:48:10.857813: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
pybullet build time: Sep 22 2020 00:55:20
startThreads creating 1 threads.
starting thread 0
started thread 0
argc=2
argv[0] = --unused
argv[1] = --start_demo_name=Physics Server
ExampleBrowserThreadFunc started
X11 functions dynamically loaded using dlopen/dlsym OK!
X11 functions dynamically loaded using dlopen/dlsym OK!
libGL error: MESA-LOADER: failed to open swrast: /home/hefeiwang/miniconda3/envs/ravens/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
Creating context
Failed to create GL 3.3 context ... using old-style GLX context
Failed to create an OpenGL context
```
参考[CSDN博客](https://blog.csdn.net/peng_258/article/details/132500323)\\
在虚拟环境中执行
```bash
 conda install -c conda-forge gcc
 ```
 问题解决，终于能正常运行$ravens/demos.py$

 不知道是不是比想不通哪里出问题但就是运行不了更伤心的是，想不通哪里出问题但问题莫名其妙自己解决了。
 运行train和test很长一段时间里都会不停循环显示
 ```bash
 Oracle demonstration:1/10
 Object for pick is not visible. 
 Skipping demonstration.Total Reward: 0 Done: True
 ```
 花了很长时间试图debug，然而后来反复运行，偶尔就能正常跑（但是正常跑过train之后再运行test依然有先失败几次，才正常）。为什么啊！难道是随机生成的方块位置没有设好边界条件，导致超出视野？

 
python 加载pkl文件
 ```bash
 python
>>>import pickle
>>> with open('../output/struct_rearrange_0/goal_spec_000004.pkl','rb') as file:
...     loaded_data = pickle.load(file)
...
>>> print(loaded_data)
```