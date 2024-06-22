import utils
import pandas as pd
df = pd.read_csv('/home/hyt/project/fake_news_detect/llm_based/result/tmp/content_based_fake.csv')
print(utils.evaluate(df))