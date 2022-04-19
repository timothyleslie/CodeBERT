## 1.25 
遇到了assertion error，说明input-ids超过了max-length。
codeT5做分类任务需要输入什么features

## 1.26
跑通codeT5+prompt的测试了，关键在于模型输入的时候需要一个decoder_input_ids的feature
在test文件里比较了t5-small和t5-base，都跑了20个epoch，发现small的效果更好一点，所以就用small来作为模型了。

## 2.8
跑完codeT5+prompt的测试，python mrr: 0.5967964600174649

## 2.10
正在跑codeT5+prompt的java测试。

尝试解决超过max-seq-length的bug。
tokenizer的encode和tokenize函数实现的是不一样的效果。
https://zhuanlan.zhihu.com/p/366911389

## 2.13
计算t5+prompt的java mrr值，出现了错误。
bug1：处理第13个batch时出现如下错误：
  File "mrr.py", line 30, in main
    correct_score = float(batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
IndexError: list index out of range
尝试把13个batch删了再重跑。
应该是12号有问题，误删了13号。

出现了系统bug，watchdog一直在报告bug。


## 3.4
参考T5中的glue QQP，https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py

我可以将输入转化成如下格式
{
       "inputs": (
           "code: Why do I easily get bored of my friends? query: Why do I get bored of my friends so quickly?"
       ),
       "targets": "relevant",
      "idx": 10,
}
代码已经放在t5-utils里面的glue函数里，接下来就是把glue函数改到convert-to-feature里。


## 4.5
目前补了hard-prompt的java、php数据集。
java数据集的测试还没有跑。
pytho数据集的测试还没跑完。(在hard-php窗口里面)
接下来跑go数据集（在train窗口里面）

## 4.7
python数据集的测试跑完。
接下来跑java数据集的测试（在eval窗口里面）
go数据集的训练跑完。
下来跑ruby数据集的训练。（在train窗口里面）

ruby数据集的训练跑完。
接下来跑JavaScript数据集的训练（在train窗口里面）

## 4.8
hard-prompt JavaScript数据集训练跑完。
接下来跑mixed-prompt ruby数据集的训练（在train窗口里面）

go的训练要迁移到/data目录下
python的测试要迁移到/data目录下
test/python/hard-prompt1也要迁移到/data目录下
上述数据迁移都已完成。

midxed-prompt ruby的训练跑完。
接下来跑mixed-prompt JavaScript的训练。


## 4.9
mixed-prompt JavaScript训练跑完。
接下来跑mixed-prompt php的训练

mixed-prompt全部数据集都已经跑完。
接下来跑hard-prompt的ruby、go、JavaScript测试，分别在eval-XX窗口中
同时还有之前java的测试还没跑完。

之前再测t5-java的mrr值时机器老是出bug，现在把它下载到本地测试一下。

跑t5small+mixedpropmt在JavaScript数据集的训练（在train窗口）
跑t5small+mixedpropmt在go数据集的训练（在train-go窗口）

## 4.10
接下来跑t5small+mixedprompt的go的测试（在eval窗口）
跑t5small+mixedprompt的JavaScript的测试（在eval-JavaScript窗口）

接下来跑t5small+mixedprompt的php训练（在train窗口）
跑t5small+mixedprompt的ruby训练（在train-ruby窗口）

## 4.13
已经把t5+mixedprompt的所有训练和测试跑完。
接下来跑bert+mixed的所有测试。其中php在eval窗口里。

## 4.14
看一下这个tensorboard是什么原因。
写一下st的训练代码。

## 4.15
排查了一下tensorboard的原因，把tr-loss=0这句代码删掉就没问题了。
st的训练代码已经写好了。

跑了ruby的训练代码。
接下来跑JavaScript的训练代码和ruby的测试代码。