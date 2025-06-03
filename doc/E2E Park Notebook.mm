
<map>
  <node ID="root" TEXT="E2E Park Notebook">
    <node TEXT="主要创新点" ID="4dbdc101d30d76de08f0bb1465009136" STYLE="bubble" POSITION="right">
      <node TEXT="单模型构架，而非拆分成流水线" ID="fc2cb7e7879199d9c54a56084c408e9a" STYLE="fork">
        <node TEXT="优势：" ID="56b3dae62e6178e050556573d2271ca5" STYLE="fork">
          <node TEXT="数据集好处理，也更方便收集（不需要拆分成数段）" ID="f5f7c88cdd855582545302aa05fcb779" STYLE="fork"/>
          <node TEXT="训练完可复现性高很多" ID="c04b404166d797bb5075f9a824bf43d6" STYLE="fork"/>
        </node>
        <node TEXT="劣势：" ID="df7ce87a7f1544f681a5aef1dfc0b64c" STYLE="fork">
          <node TEXT="数据需求可能会变大" ID="ccf2739f4b878a5fc5a75a39f8977099" STYLE="fork"/>
        </node>
        <node TEXT="特点：" ID="d362705af1548fca24a3d4f09eec9ec4" STYLE="fork">
          <node TEXT="2种 encoder并行，并组合成BEV 表征" ID="1da7e26b9b3ac3bd8e0f7e8affe31467" STYLE="fork">
            <node TEXT="BEV 表征处于核心位置" ID="9c1a424b857f3ee9eef1cdb154a17aa3" STYLE="fork"/>
            <node TEXT="摄像头信息(Camera encoder) " ID="ece1280036d5f54037283f914046ad6e" STYLE="fork">
              <node TEXT=" EfficientNet做基础特征提取" ID="67cfe90895c808b9defed70e026bb81c" STYLE="fork"/>
              <node TEXT="升维(学习depth的分布(d_dep) 信息，特征最终为3D）" ID="61e18e1c34e2ff62c6577bb7261c2843" STYLE="fork"/>
            </node>
          </node>
          <node TEXT="一个特殊的Q(车位 embedding，包含position embedding),KV(BEV embedding，包含position embedding) cross attention，作为关键检索条件" ID="45611bd59f570a7d68a9b3fb390349d5" STYLE="fork"/>
          <node TEXT="基于自回归构建E2E轨迹图, 基于transformer decoder" ID="3c0374a7ea400eba91a02c9a6cae5e63" STYLE="fork">
            <node TEXT=" K,V 同 encoder，Q 则为已生成个轨迹点" ID="eddaaaf5792be32e110f06f01e8778f6" STYLE="fork"/>
            <node TEXT="上下文信息依赖BEV的全局注意力机制" ID="bb6399326cbbf0f1661ccacf7b78c7e7" STYLE="fork"/>
          </node>
        </node>
      </node>
      <node TEXT="多摄像头转BEV(Bird’s Eye View)" ID="242be1221142c0cde35efcafd4da6720" STYLE="fork">
        <node TEXT="motivation: BEV 优势极大" ID="c33b779f29210fabad55335f9f3b056b" STYLE="fork">
          <node TEXT="物理可解释性" ID="1a35397209ffd63164f4d63c556b3cfa" STYLE="fork"/>
          <node TEXT="路径规划直观" ID="afb86803af4158147cbb5694584c9913" STYLE="fork"/>
          <node TEXT="图语义分割明确" ID="779c839493c5b0cc6fc578850f38d92c" STYLE="fork"/>
          <node TEXT="几何上更可验证" ID="fff1c2e30642ecf83a9ee2005de0a9f0" STYLE="fork"/>
          <node TEXT="简化任务-&gt; 从image navigation 转换为了 waypoint navigation" ID="12d7152d7f1b8513b70891cfb07e3c33" STYLE="fork"/>
        </node>
        <node TEXT="Note:" ID="2f2c7b9018ad019c80ba12c455a87794" STYLE="fork">
          <node TEXT="思路接近于先构建3D重建的SLAM, 并转换SLAM 视角，进行路径规划" ID="8f4c6e34c1820cc53cd42a4f779efbe8" STYLE="fork"/>
        </node>
        <node TEXT="modelling(BevEncoder):" ID="e2599340d3d4208a57e14a939ae71cb5" STYLE="fork"/>
      </node>
    </node>
    <node TEXT="问题及数据集定义" ID="16996d07096f788ccbb7a88a3ccde852" STYLE="bubble" POSITION="right">
      <node TEXT="\mathcal{D}=\{(I_{i,j}^k, P_{i,j}, S_i)\}， I -&gt; Image, P -&gt; trajectory way point, S -&gt; Target Slot" ID="49902c9ac2cfe775498183d8e91c652f" STYLE="fork">
        <node TEXT="数据集很接近image navigation" ID="b09ead5780cb48e4dcd2aa9c89aa3185" STYLE="fork">
          <node TEXT="input : 4张图+goal(x,y,z,yaw)" ID="b9f13ddbc94866cfde5b8995c5436dc6" STYLE="fork"/>
        </node>
      </node>
    </node>
    <node TEXT="未解决的问题：" ID="f3845e3041ec6970654e9c021c22e954" STYLE="bubble" POSITION="right">
      <node TEXT="input channel 为什么是64" ID="564d2c4d7032f20afd655d5273464b97" STYLE="fork"/>
      <node TEXT="绝对坐标和相对坐标（包括深度）的转换融合，是在哪个layer学习的，" ID="c036be98bd913cfbe2c83a678ed67ae3" STYLE="fork"/>
      <node TEXT="分析Bag2E2E 这个function，是潜在的数据集构建的基座方法 -&gt; 需要绍谦协助" ID="9a8a693f788e8ffa917a024971f7b9f9" STYLE="fork">
        <node TEXT="涉及摄像头数据处理" ID="7b0a24222615748389c021d00139cd5c" STYLE="fork"/>
        <node TEXT="复杂度很高" ID="323516e2ca740bdf132001312510808c" STYLE="fork"/>
        <node TEXT="可以作为长期对齐目标：做一个拓扑结构的Bag2E2E" ID="ce586dac5758572e3fa54f0ab9b4dbf2" STYLE="fork"/>
      </node>
    </node>
    <node TEXT="Float 有精度问题，绝对值1e-2以下的值全部取0" ID="090a686c91b231d65ef08b25d8ba6a42" STYLE="bubble" POSITION="right"/>
  </node>
</map>