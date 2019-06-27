# Anime Character Classifier

## 关于

该项目是人工智能课程的期末作业，故只是实现一个 Demo，该项目将不会再深入或更新。

采用的是 Keras + Tensorflow，数据集是自己做的，是个很小的数据集，但是效果还可以，通过在训练上的一些 Trick 最终模型准确度在97%-98%这样，项目将提供一个模型的下载。

模型只提供一个，而且这个模型存在一定的问题，由于数据集的图片不太全面，对一些角色处于倒立状态的图片不太能识别。

## 用法

提供了完善的训练、再训练、测试、评估的代码，具体的用法可以用-h参数查看，一些参数仍然需要在源码中修改。

你可以通过修改神经网络的结构、完善数据集等方式来提升识别的准确率。

## 免责声明

该项目数据集中的图片均通过搜索引擎搜索、下载得到或直接从相关动漫中截图、抽帧，未获得任何相关的版权授权，故该项目及训练集中的图片仅供用于学习用途。

## 相关下载

[数据集 (96.18MB, dataset.7z)](http://smallfile.backrunner.top/AnimeCharacterClassifier/dataset.7z)

SHA1: AEEDD90CB3633AC24EB080D6BC88F7B4A204BDF5

[模型 (154.22MB, models.7z)](http://smallfile.backrunner.top/AnimeCharacterClassifier/models.7z)

SHA1: 262A63AA3520D12D825EC5CD47A621E71613602B

## Tips

如果你想基于这个项目完善并做成一个网站，可直接调用 test.py，它输出的是一个可解析的JSON。