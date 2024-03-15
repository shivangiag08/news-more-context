import pandas as pd

bbc_data = pd.read_csv("C:\Shivangi\college\Sem 4\MLPR\project/news-more-context\output_10k.csv")
bbc_data = bbc_data.sample(n=200)

bbc_data.to_csv("C:\Shivangi\college\Sem 4\MLPR\project/news-more-context\output_200.csv", index=False)