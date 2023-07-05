# ForexPrediction

基于 GluonTS 的汇率预测。(一个尝试)

**本项目的预测结果的准确性低，任何因本项目直接或间接导致的损失，本项目不承担任何责任。**

### 数据来源
[Investing.com](https://www.investing.com/currencies/usd-cny-historical-data)

将数据放到 /data 下。
运行 crawler.py 更新数据。
运行 train.py 训练，模型保存在 /model
运行 forcast.py 预测