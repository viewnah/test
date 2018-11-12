# Tensorflow Lite使用教程

Tensorflow Lite是在移动和嵌入式设备上运行机器学习模型的官方解决方案，支持Android和ios。本教程会指导你如何从头开始开发一个手写数字识别android app。本教程完整参考了开源项目[tflite-mnist-android](**tflite-mnist-android**)

**Requirements:**

* Python 3.6
* Tensorflow 1.12.0
* Android Studio 3.2

## 模型文件的生成

首先利用tensorflow构建一个手写数字识别的卷积神经网络模型，模型的构建不是本教程的重点，详细内容可以参考[tensorflow官方教程](https://www.tensorflow.org/tutorials/)

tensorflow训练完成后可以根据需要生成以下几种模型文件：

* GraphDef
* CheckPoint
* FrozenGraphDef
* SavedModel
* Tensorflow lite model

GraphDef一般后缀名为.pb，保存了图模型的计算流程，包括了图中的常量，但不保存变量,可以使用以下方式获取：

```python
import tensorflow as tf

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	tf.train.write_graph(sess.graph_def, './model_dir/mnist.pb', as_text=False)
```

CheckPoint保存了图模型中变量的值，在旧版tensorflow中保存为后缀名为.ckpt的文件。在新版tensorflow中会保存成四种文件类型，分别为.meta、.index、.data和checkpoint。.meta文件保存了网络的图结构，包含变量、op、集合等信息；.data和.index保存了网络中所有权重、偏置等变量数值；checkpoint文件记录了最新保存的模型文件列表。保存方式如下：

```python
import tensorflow as tf

saver = tf.train.Saver()
with tf.Session() as sess:
    ...
	saver.save(sess, "./model_dir/mnist")
```

FrozenGraphDef依然为后缀名为.pb的模型文件，不仅保存了图模型的计算流程和常量，而且将模型中所有变量都转变为常量。保存方式如下：

```python
import tensorflow as tf
from tensorflow import graph_util

with tf.Session() as sess:
	graph_def = tf.get_default_graph().as_graph_def()
	frozen_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ["output"])  # output为输出tensor的名称
	with tf.gfile.GFile('./model_dir/mnist.pb', "wb") as f:
		f.write(frozen_graph_def.SerializeToString())
```

SavedModel是一种与语言无关，可恢复的密封式序列化格式。官网的解释为带有签名的GraphDef和CheckPoint，标记了模型的输入和输出参数，可以从SavedModel中提取出GraphDef和CheckPoint。保存方法如下：

```python
import tensorflow as tf

export_dir = "MetaGraphDir/mnist"
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
    builder.add_meta_graph_and_variables(sess,
                                         ["TRAINING"],
                                         signature_def_map=foo_signatures,
                                         assets_collection=foo_assets
                                         )
# Add a sencod MetaGraphDef for inference
with tf.Session(graph=tf.Graph()) as sess:
    builder.add_meta_graph(["SERVING"])
builder.save()
```

Tensorflow lite model既为本教程所需要的移动端模型文件，一般保存为后缀名为.tflite或.lite的文件。生成.tflite文件有两种方式：

第一种方法，首先保存图模型文件（GraphDef）和变量文件（CheckPoint），然后利用freeze_graph工具生成FrozenGraphDef文件，最后利用toco工具生成tflite文件。具体过程如下：

1. **保存GraphDef和CheckPoint**

   ```python
   saver = tf.train.Saver()
   
   with tf.Session() as sess:
   	sess.run(tf.global_variables_initializer())
   	tf.train.write_graph(sess.graph_def, "model/", "mnist.pb", as_text=False)
   	saver.save(sess, "model/mnist")
   ```

   当模型训练完后，model文件夹中会保存以下模型文件：

   - checkpoint
   - mnist.data-00000-of-00001
   - mnist.index
   - mnist.meta
   - mnist.pb

2. **ubuntu下安装[Bazel](https://docs.bazel.build/versions/master/install.html)**

   bazel是google开源的一套编译构建工具，安装过程如下：

   1.  安装依赖

      `sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python`

   2. 下载二进制包文件`bazel-<version>-installer-linux-x86_64.sh`,地址:[Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases)

   3. 安装sh文件

      ```shell
      chmod +x bazel-<version>-installer-linux-x86_64.sh
      ./bazel-<version>-installer-linux-x86_64.sh --user
      ```

   4. 设置环境

      `export PATH="$PATH:$HOME/bin"`

3. **克隆Tensorflow仓库**

   `git clone https://github.com/tensorflow/tensorflow`

4. **编译freeze_graph、toco、summarize_graph工具**

   ```shell
   cd tensorflow
   
   bazel build tensorflow/python/tools:freeze_graph
   bazel build tensorflow/contrib/lite/toto:toto
   Bazel build tensorflow/tools/graph_transforms:summarize_graph
   ```

5. **使用summarize_graph查看模型网络结构，找到自己模型的输入和输出**

   ```
   bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
   	--in_graph=../model/mnist.pb
   ```

   in_graph为GraphDef模型保存位置

   实际情况下由于模型的结构和名称往往比较复杂，建议在模型构建时为输入和输出tensor设置好name参数

6. **使用freeze_graph将GraphDef和CheckPoint固化成FrozenGraphDef**

   ```shell
   bazel-bin/tensorflow/python/tools/freeze_graph\
           --input_graph=../model/mnist.pb \
           --input_checkpoint=../model/mnist \
           --input_binary=true \
           --output_graph= ../model/frozen_mnist.pb \
           --output_node_names=output
   ```

   input_graph为GraphDef模型保存位置

   input_graph为CheckPoint模型保存位置，实际读取的是mnist.data-00000-of-00001，只需要指定mnist文件名即可

   input_binary表明读写均为二进制格式

   output_graph为FrozenGraphDef模型保存位置

   output_node_names为输出节点的名称，需要在训练模型时指定

   当然，像之前所说，我们可以从代码中直接保存FrozenGraphDef

7. **使用toco将FrozenGraphDef转换为tflite**

   ```shell
   bazel run --config=opt tensorflow/contrib/lite/toco/toco \
   	--input_file=../model/frozen_graph.pb \
   	--input_format=TENSORFLOW_GRAPHDEF \
   	--output_format=TFLITE \
   	--output_file=../model/mnist.tflite \
   	--inference_type=FLOAT \
   	--input_type=FLOAT \
   	--input_arrays=input \
   	--output_arrays=output \
   	--input_shapes=1,28,28,1
   ```

   input_file为FrozenGraphDef文件位置

   input_format待转换的模型文件类型

   output_format转换后的模型文件类型

   output_file为tflite模型保存位置

   inference_type为模型推理时的数据类型

   input_type为模型输入的数据类型，和inference_type都为FLOAT，除非你要生成quantized model

   input_arrays为模型中输入节点的名称

   output_arrays为模型中输出节点的名称

   input_shapes为模型输入的维度

<hr>

第二种方法是使用代码来生成tflite，由于各种原因在编译时很容易出现bug，所以比较推荐第二种方法，官方给出的代码如下所示：

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
out = tf.identity(val, name="out")

with tf.Session() as sess:
  tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [img], [out])
  open("converteds_model.tflite", "wb").write(tflite_model)
```

这里的代码可以说明我们可以直接使用graph_def和输入输出节点直接生成tflite文件，但这个例子中由于不存在变量所以GraphDef无需冻结成FrozenGraphDef，实际情况下我们需要先生成FrozenGraphDef：

```python
frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, 	['output'])  #这里 ['output']是输出tensor的名字
tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [input], [out])   #这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
open("model.tflite", "wb").write(tflite_model)
```

表面上看来已经解决了所有问题，但在我们搭建模型的过程中，我们往往会这样设置input：

`input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')`

注意这里input的batch_size为None，没有指定为具体的数值，这时我们使用tf.contrib.lite.toco_convert时就会报错，在实际使用中还需要先固定batch_size的维度，使用起来比较复杂。在最新的tensorflow1.12.0版本中toco_convert函数已经被设为deprecated，并且官方推荐我们使用TFLiteConverter，不得不吐槽官方教程没有及时更新。实际使用起来TFLiteConverter的确更加方便,不仅不需要考虑batch_size,而且输入输出是tensor本身，不是其名称。TFLiteConverter支持从session、FrozenGraphDef、SavedModel和keras模型进行转换，官方教程如下：

```python
from tensorflow.contrib import lite

converter = lite.TFLiteConverter.from_session(sess, in_tensors, out_tensors)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# Converting a GraphDef from file.
converter = lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# Converting a SavedModel.
converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Converting a tf.keras model.
converter = lite.TFLiteConverter.from_keras_model_file(keras_model)
tflite_model = converter.convert()
```

现在让我们训练自己的模型：

`python train.py --model_dir ./saved_model --iterations 10000`

我们已经得到mnist.tflite文件，接下来需要在安卓端使用该模型文件

## 安卓端模型使用

首先，使用Android Studio新建一个项目，在app下build.gradle中的dependencies里添加

`implementation 'org.tensorflow:tensorflow-lite:1.10.0'`

在java类中这样导入

`import org.tensorflow.lite.Interpreter;`

将mnist.tflite放入`app/src/main/assets`文件夹下，官方给出如下函数代码读取该模型文件：

```java
private static final String MODEL_PATH = "mnist.tflite";

private MappedByteBuffer loadModelFile(Activity activity) throws IOException{
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declareLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declareLength);
    }
```

在类中读取：

```java
import org.tensorflow.lite.Interpreter;

public Classifier(Activity activity) throws IOException{
        mInterpreter = new Interpreter(loadModelFile(activity));
```

模型的输入使用ByteBuffer数据类型，模型的输出在本例中是一个1行10列的二维数组，存储了10个数字的预测概率,使用Interpreter.run即可进行推理

```java
private final ByteBuffer mImgData;
private final float[][] mResult = new float[1][10];
...
mInterpreter.run(mImgData, mResult);
```

使用输出的时候需要自己编写argmax函数，详见项目代码Result.java

原作者使用了自己编写的FingerPaint包，在build.gradle中dependencies下添加

`implementation 'com.nex3z:finger-paint-view:0.1.0'`

然后就可以在Containers/view中找到FingerPaintView

mnist数据实际上是0-1的浮点数，颜色越深越靠近1，与实际图像刚好相反，所以需要进行反色处理，详见代码ImageUtil.java,另外还需要将图像变为灰度图，并缩放到0-1区间，详见代码Classifier.java

编译app时，需要在build.gradle中android下添加

```
    aaptOptions {
        noCompress "tflite"
        noCompress "lite"
    }
```

避免模型被压缩，因而无法加载

安卓开发的过程不在此教程进行说明，最后试试这个软件吧



![img](assets/demo.gif)