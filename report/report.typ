#import "template/template.typ": *

#show: project.with(
  title: "《计算机图形学》项目报告",
  name: "三维俄罗斯方块实验",
  id: "23060827",
  class: "计算机科学英才班",
  authors: "鲍溶",
  department: "卓越学院",
  date: (2025, 06, 09),
  cover_style: "hdu_report",
)

#toc()
#pagebreak()

= 实验要求与目的

课程要求独立设计一个计算机动画演示系统，需要包含如下核心技术中的1至3项，其中4和5项必须至少涉及1点：

+ 多个三维图形元素（如六面体，球）；
+ 包含多个或者多种光源的光照效果；
+ 颜色纹理或者几何纹理；
+ 鼠标、键盘等交互编辑功能；
+ 包含旋转、变形等动画元素（随时间变化）。

本实验要求综合运用课程中学习的内容，通过OpenGL编程实现包含多个功能的动画系统，以检验学生的理论学习情况与编程能力。

= 概要设计

实验项目为“三维俄罗斯方块游戏”，包含俄罗斯方块的基础下落式玩法，同时添加一定的辅助功能，以提升三维化后游戏的可玩性。游戏程序需要处理基本游戏逻辑、自由视角移动、用户游戏操作输入、简单光照模型、过渡动画播放等行为。

#img(
  image("./assets/project.svg"),
  caption: "程序功能模块框图",
) <figure:arch>

如@figure:arch 所示，程序主要包含四类功能模块：主要包含抽象游戏逻辑的模型（图中红色矩形）、主要控制绘制与渲染的视图（图中绿色矩形）、协调上述两类模块的控制器（图中黄色矩形）、用于完成输入输出的基础库（图中青色矩形）。

程序采用基于事件的交互设计。控制器注册回调函数，接收来自基础库的用户输入事件与定时器过期事件。根据游戏逻辑，控制器按固定间隔请求更新游戏模型，并在有需要时触发游戏区域的重绘操作。模型的抽象数据通过视图模块转换为几何体信息，并进入相应的渲染器进行渲染。根据用户输入，当需要播放过渡动画时，控制器向动画引擎发出入队申请，将动画事件推入队列中。动画引擎根据定时器信息调度当前正在播放的动画事件，根据相应的渐变函数计算进度，并执行事件中指定的操作。

= 详细设计

== 语言与基础库选择

项目使用Python语言实现。程序使用PyOpenGL进行图形编程，该库是OpenGL API的Python binding。矩阵库选择NumPy库，通过自动向量化提升运算效率#cite(label("harris2020array"))。使用Pillow库#cite(label("andrew_murray_2025_15204076"))读取BMP格式纹理。用户输入与窗口控制通过GLUT实现，以简化复杂的操作系统交互设计。

== 抽象游戏逻辑

项目游戏逻辑基于常规俄罗斯方块游戏规则#cite(label("plank2022fifty-tetris"))简化、扩展而来。基本而言，游戏将在游戏区域顶端生成随机形状、朝向的四联骨牌（Tetromino），其会随游戏进行而下落。玩家需要通过旋转、平移方式将其放置于游戏区域底部，当一层的方块被完全填满后，该层将消去，玩家即得分。当因未填满的层达到游戏区域顶端，致使无法生成新的骨牌时，游戏结束。为了程序编写方便，项目中不包含原规则书中描述的“Hold”玩法；为了三维游戏可玩性，项目对规则书中的部分规则进行了增广或修改。

=== 自由骨牌集

如@figure:sided-tetrominos 所示的骨牌组成最小单面骨牌集。这些骨牌在旋转、平移操作下互不全等。在常规俄罗斯方块规则中，骨牌可以旋转、平移，但是无法镜像。因此，最小单面骨牌集为常规俄罗斯方块游戏中最小可玩骨牌集。在三维俄罗斯方块游戏中的旋转操作不需要如此强的约束条件，因此如图@figure:free-tetrominos
所示的最小自由骨牌集即可完成游戏。

#img(
  image("assets/2d-sided-tetrominos.svg", width: 80%),
  caption: "最小单面骨牌集",
) <figure:sided-tetrominos>

#img(
  image("assets/2d-free-tetrominos.svg", width: 30%),
  caption: "最小自由骨牌集",
) <figure:free-tetrominos>

=== 规则扩展

墙踢（wall kick）指当骨牌的旋转方向中的少数几个被遮挡时可能采取的旋转行为。如@figure:wall-kick-example 所示，在该二维俄罗斯方块游戏场景中，左上角的T型骨牌无法在不超出游戏区域的前提下继续绕轴心旋转。若不移开该骨牌，这种限制会导致其旋转死锁于该位置，影响游戏流畅性。此时即需要墙踢机制生效，自动将骨牌向右平移1格以完成旋转。

在三维俄罗斯方块中，需要沿八个方向判断墙踢调整平移的可行性。

#img(
  image("assets/wall-kick-example.png"),
  caption: "二维墙踢的适用场景",
) <figure:wall-kick-example>

== 绘制与渲染

=== 普通对象绘制

项目使用OpenGL固定功能管线渲染所有几何体。固定功能管线属于遗留功能，依赖预定义的渲染管线工作，用户仅需传入功能开关、绑定顶点信息即可实现简单的绘制功能。相比于基于自定义渲染器的可编程管线而言，固定功能管线使用简便，Blinn-Phong光照模型运算代价低廉，能够满足本项目的需求。

如@code:block-renderer 所示，项目为每一类几何体创建相应的顶点位置、纹理坐标、法线信息、颜色信息，将其绑定到顶点缓冲后，根据视图传入的几何体信息发出绘制指令。

#code(
  ```py
  class BlockRenderer:
      Vertex = tuple[VecXYZf, VecUVf, VecXYZf]  # (pos, uv, normal)

      VERTICES: list[Vertex] = [
          # Front
          ((0.0, 0.0, 1.0), (0.0, 0.0), (0.0, 0.0, 1.0)),
          ((1.0, 0.0, 1.0), (1.0, 0.0), (0.0, 0.0, 1.0)),
          ((1.0, 1.0, 1.0), (1.0, 1.0), (0.0, 0.0, 1.0)),
          ((0.0, 1.0, 1.0), (0.0, 1.0), (0.0, 0.0, 1.0)),
          ...
          # Left
          ((0.0, 0.0, 1.0), (1.0, 0.0), (-1.0, 0.0, 0.0)),
          ((0.0, 1.0, 1.0), (1.0, 1.0), (-1.0, 0.0, 0.0)),
          ((0.0, 1.0, 0.0), (0.0, 1.0), (-1.0, 0.0, 0.0)),
          ((0.0, 0.0, 0.0), (0.0, 0.0), (-1.0, 0.0, 0.0)),
      ]

      _texture_id: int
      _vertex_vbo: vbo.VBO

      def __init__(self, texture_id: int):
          vertices_data = np.array(
              [
                  v
                  for vertex in self.VERTICES
                  for v in it.chain(
                      (
                          min(1.0 - RENDER_BLOCK_GAP / 2, max(RENDER_BLOCK_GAP / 2, v))
                          for v in vertex[0]
                      ),
                      vertex[1],
                      vertex[2],
                  )
              ],
              dtype=np.float32,
          )

          self._vertex_vbo = vbo.VBO(vertices_data)
          self._texture_id = texture_id

      def render(self, block: BlockView):
          pos = block.pos

          gl.glPushMatrix()
          gl.glTranslate(pos[0], pos[2], pos[1])

          gl.glColor4f(*block.color, block.alpha)
          gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, MATERIAL_AMBIENT)
          gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, MATERIAL_DIFFUSE)
          gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, MATERIAL_SPECULAR)
          gl.glMaterialfv(gl.GL_FRONT, gl.GL_SHININESS, MATERIAL_SHININESS)

          gl.glEnable(gl.GL_TEXTURE_2D)
          gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
          self._vertex_vbo.bind()
          gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
          gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
          gl.glEnableClientState(gl.GL_NORMAL_ARRAY)

          gl.glVertexPointer(3, gl.GL_FLOAT, 32, self._vertex_vbo)
          gl.glTexCoordPointer(2, gl.GL_FLOAT, 32, self._vertex_vbo + 12)
          gl.glNormalPointer(gl.GL_FLOAT, 32, self._vertex_vbo + 20)
          gl.glDrawArrays(gl.GL_QUADS, 0, len(self.VERTICES))

          gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
          gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
          gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
          self._vertex_vbo.unbind()
          gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
          gl.glDisable(gl.GL_TEXTURE_2D)

          gl.glPopMatrix()
  ```,
  caption: [方格渲染器#raw("BlockRenderer")的实现，其中顶点信息已经过缩减],
  label: <code:block-renderer>,
)

=== 混合绘制透明与不透明对象

将透明对象与不透明对象一同绘制时，若不透明面处于透明面后，深度测试将错误地将其消隐。因此，透明物体不应该参与深度测试。使用@code:mixed-obj 所示的两遍算法能够较好解决该问题，首先正常绘制不透明对象，再在关闭深度测试后绘制透明面。这样可以正确处理透明面叠加的情况，避免透明面错误遮挡后方的其他面片。

#code(
  ```txt
  enable depth test
  for obj in filter(not is_transparent, objects):
      render(obj)
  disable depth test
  for obj in filter(is_transparent, objects):
      render(obj)
  enable depth test
  ```,
  caption: "两遍混合透明度对象渲染",
  label: <code:mixed-obj>,
)

== 用户交互

=== 相对视角移动

在所有第一人称游戏中，摄像机的平移方向均相对于视线方向计算。在本项目中，为了操作便利，同样采用相对视角移动机制。如@code:eye-rel-movement 所示，通过复用摄像机偏航（yaw）参数，可以低成本地将摄像机坐标系映射到世界坐标轴上。

#code(
  ```py
  def get_move_dir(yaw: float, key: ty.Literal[b"w", b"a", b"s", b"d"]) -> MoveDir:
      yaw_ = round(yaw)
      if yaw_ >= 315 or yaw_ < 45:
          if key == b"w":
              return MoveDir.Z_NEG
          elif key == b"a":
              return MoveDir.X_NEG
          elif key == b"s":
              return MoveDir.Z_POS
          elif key == b"d":
              return MoveDir.X_POS
      elif 45 <= yaw_ < 135:
          if key == b"w":
              return MoveDir.X_NEG
          elif key == b"a":
              ...
          ...
      elif 135 <= yaw_ < 225:
          if key == b"w":
              return MoveDir.Z_POS
          elif key == b"a":
              ...
          ...
      else:
          if key == b"w":
              return MoveDir.X_POS
          elif key == b"a":
              ...
          ...
  ```,
  caption: "获取相对视角的X-Z平面移动方向",
  label: <code:eye-rel-movement>,
)

=== 屏上点选

在传统二维俄罗斯方块游戏中，玩家随时都能观察到每一层是否完全被填满，因此能够不间断地规划下一个骨牌该摆放至何位置。在本程序中，每一Y轴层上方格有很大概率会互相遮挡，导致玩家难以观察到空格。屏上点选功能允许玩家在暂停状态下使用鼠标选择某个格子，按照指定的坐标轴高亮该格子所在平面，同时淡化所有其他格子，以更好观察该层的摆放情况。

实现该功能需要找到一个鼠标点击位置下方最近的一个方格。将一条正交于视口平面、长度为1的线段逆投影变换至世界坐标系中，再沿该线段方向发出射线，与该射线相交的第一个方格即为待求方格。由于游戏中所有方格均沿坐标系正交排列，存在高效算法能够快速计算该结果#cite(label("10.1145/15886.15916"))。该算法的伪代码形如@code:aabb-algo。

#code(
  ```txt
  def slab_method(l, h, o, r):
      tl = (l - o) / r
      th = (h - o) / r
      tc = [min(tl_i, th_i) for tl_i, th_i in zip(tl, th)]
      tf = [max(tl_i, th_i) for tl_i, th_i in zip(tl, th)]
      tcm = max(tc)
      tfm = min(tf)
      return tcm <= tfm and tcm >= 0
  ```,
  caption: "Slab法射线追踪碰撞检测",
  label: <code:aabb-algo>,
)

== 动画系统

传统俄罗斯方块游戏中，各关键帧之间不存在补间动画，这样“像素式”的艺术风格也受到大多数玩家喜爱。但是，诸如屏上点选等辅助功能的引入，使得纯离散式的动画风格不再适用。为了达到视觉上的和谐感，程序为屏上点选功能加入动画引擎，实现任意函数插值的补间效果。

如@code:ani-engine 所示，程序实现了一个简易的事件驱动动画引擎。在每个定时器事件发出时，动画引擎都会刷新当前动画组中各动画的状态与进度。用户可以向当前动画组中推入新的动画，也可以选择结束当前动画组。当且仅当一个动画组的前驱内所有动画都播放完毕后，下一个时刻，该动画组才会开始播放。

#code(
  ```py
  class AnimationEngine:
      _queue: list[list[Animation]]

      def __init__(self):
          self._queue = []

      def split_keyframe(self):
          self._queue.append([])

      def fire(self, animation: Animation):
          if len(self._queue) == 0:
              self.split_keyframe()
          self._queue[-1].append(animation)

      def tick(self, elapsed_us: int):
          if len(self._queue) == 0:
              return
          for animation in self._queue[0]:
              animation.tick(elapsed_us)
          if all(animation.done for animation in self._queue[0]):
              self._queue.pop(0)
  ```,
  caption: "简易事件队列型动画引擎实现",
  label: <code:ani-engine>,
)

= 实验结果

@figure:gameplay-0、@figure:gameplay-1 和@figure:gameplay-2 展示了游戏的游玩界面与定位辅助线功能。@figure:gameplay-xray-0、@figure:gameplay-xray-1 和@figure:gameplay-xray-2 分别展示了不同透视模式下不同角度观察到的场景。

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [#img(
      image("./assets/gameplay-0.png"),
      caption: "某局游戏的截图",
    ) <figure:gameplay-0>
  ],
  [#img(
      image("./assets/gameplay-1.png"),
      caption: "开启定位辅助线后的同个场景",
    ) <figure:gameplay-1>
  ],
  [#img(
      image("./assets/gameplay-2.png"),
      caption: "另一个视角下的同个场景",
    ) <figure:gameplay-2>
  ],

  [#img(
      image("./assets/gameplay-xray-0.png"),
      caption: "开启Y轴透视模式后的同个场景",
    ) <figure:gameplay-xray-0>
  ],
  [#img(
      image("./assets/gameplay-xray-1.png"),
      caption: "X轴透视模式下的同个场景",
    ) <figure:gameplay-xray-1>
  ],
  [#img(
      image("./assets/gameplay-xray-2.png"),
      caption: "另一个视角下的X轴透视场景",
    ) <figure:gameplay-xray-2>
  ],
)

= 源代码

本实验与本报告的源代码可以在#link("https://github.com/CSharperMantle/hdu2025_graphics/")处获取。程序源代码按照Apache 2.0许可协议授权。

#pagebreak()

#bibliography("bib.bib", style: "gb-7714-2015-numeric")
