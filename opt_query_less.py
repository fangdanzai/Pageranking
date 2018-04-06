import numpy as np
import pandas as pd
import datetime

from sklearn import preprocessing

def jaccard(a,b):
    return 1.0 * len(a.intersection(b)) / len(a.union(b))

def minmaxscale(X):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(X)

def create_output(result, train_queries, test_queries):
        tset = set(train_queries["queryId"].values)
        result_train = result.loc[result["queryId"].isin(tset)]
        to_ranklib(result_train, "../data/output/opt_query_less_train.txt")

        tset = set(test_queries["queryId"].values)
        result_test = result.loc[result["queryId"].isin(tset)]
        to_ranklib(result_test, "../data/output/opt_query_less_test.txt")
        result_test[["queryId","itemId"]].to_csv("../data/output/opt_query_less_test_ids.txt", index=False)


def to_ranklib(df, filename):

    outfile = open(filename, "w")

    for row in df[["queryId","itemId","clicked","viewed","purchased","rank","rank_size","itemView","itemClick",\
        "itemPurchase","itemMeanRank","itemMaxRank","itemMinRank","itemMedianRank", "pricelog2",
        "maxp","minp","medianp","meanp","rangep","howexpensive","catperc","count_clicks", "uniqueUserView",
        "uUserView", "uUserPurchase", "avg_ndcg", "avg_duration", "mmr_view", "productNameSize",  "ndcgPerSession",
        "durationPerSession"]].itertuples():

        relevance = 0
        if row[3] or row[4]:
            relevance = 1
        elif row[5]:
            relevance = 2

        qid = row[1]

        outfile.write("%d qid:%d" % (relevance, qid))
        for i, v in enumerate(row[6:], start=1):
            outfile.write(" %d:%.4f" % (i,v))
        outfile.write("\n")
    outfile.close()

all_queries = pd.read_csv('../data/input/train-queries.csv', sep=';')
all_queries["eventdate"] = pd.to_datetime(all_queries["eventdate"])

all_queries = all_queries[all_queries["searchstring.tokens"].isnull()]

query_item = []
for query, items in all_queries[["queryId", "items"]].values:
    items = map(np.int64,items.split(','))
    for i in items:
        query_item.append( (query, i) )
query_item = pd.DataFrame().from_records(query_item, columns=["queryId","itemId"])

item_views = pd.read_csv('../data/input/train-item-views.csv', sep=';')
item_views.sort_values(["sessionId", "userId", "eventdate", "timeframe", "itemId"], inplace=True)
print('Item views', len(item_views))

clicks = pd.read_csv('../data/input/train-clicks.csv', sep=';')
clicks.sort_values(["queryId", "timeframe", "itemId"], inplace=True)
print('Clicks', len(clicks))

purchases = pd.read_csv('../data/input/train-purchases.csv', sep=';')
print('Purchases', len(purchases))
purchases.sort_values(["sessionId", "userId", "eventdate", "timeframe", "itemId", "ordernumber"], inplace=True)

products = pd.read_csv('../data/input/products.csv', sep=';')
print('Products', len(products))
products.sort_values(["itemId"], inplace=True)

products_category = pd.read_csv('../data/input/product-categories.csv', sep=';')
print('Products Categories', len(products))
products_category.sort_values(["itemId"], inplace=True)

query_item = pd.merge(query_item, all_queries[["queryId", "sessionId"]], how="left")
query_item = pd.merge(query_item,  clicks, how="left")
query_item.rename(columns={"timeframe":"clickTime"}, inplace=True)
query_item = pd.merge(query_item,  item_views, how="left")

query_item.rename(columns={"eventdate":"eventdateView", "timeframe":"viewTime", "userId": "userView"}, inplace=True)
query_item = pd.merge(query_item, purchases, how="left")
query_item.rename(columns={"eventdate":"eventdatePurchase", "timeframe":"purchaseTime", "userId": "userPurchase"}, inplace=True)

query_item["rank"] = 1
query_item["rank"] = query_item[["queryId","rank"]].groupby("queryId")["rank"].cumsum()
query_item["rank"] = query_item["rank"] - 1

items_per_query = query_item[["queryId","rank"]].groupby("queryId")["rank"].max()
items_per_query.name = "rank_size"

query_item = pd.merge(query_item, items_per_query.reset_index(), how="left")
query_item["rank"] = 1.0 - (query_item["rank"] / query_item["rank_size"])

query_item["clicked"] = ~query_item["clickTime"].isnull()
query_item["viewed"] = ~query_item["viewTime"].isnull()
query_item["purchased"] = ~query_item["purchaseTime"].isnull()

products_info = pd.merge(query_item[["queryId", "itemId"]].drop_duplicates(), all_queries[["queryId", "searchstring.tokens"]], on="queryId", how="left").merge(products, on="itemId", how="left")

ndcgs = pd.read_csv("../data/input/ndcg_test.csv")

train_queries = all_queries[all_queries['is.test'] == False]
test_queries = all_queries[all_queries['is.test'] == True]

train_queries.reset_index(inplace=True, drop=True)
test_queries.reset_index(inplace=True, drop=True)

all_queries = pd.merge(all_queries, ndcgs, how="left")
qstr_ndcg = all_queries[["searchstring.tokens", "qndcg"]].groupby("searchstring.tokens")["qndcg"].mean().reset_index().rename(columns={"qndcg": "avg_ndcg"})
all_queries = pd.merge(all_queries, qstr_ndcg, how="left")

qstr_duration = all_queries[["searchstring.tokens", "duration"]].groupby("searchstring.tokens")["duration"].mean().reset_index().rename(columns={"duration":"avg_duration"})
all_queries = pd.merge(all_queries, qstr_duration, how="left")

all_queries["avg_duration"].fillna(all_queries["avg_duration"].mean()/2., inplace=True)
all_queries["avg_ndcg"].fillna(all_queries["avg_ndcg"].mean()/2., inplace=True)


itemViewPopularity = item_views["itemId"].value_counts()
itemClickPopularity = clicks["itemId"].value_counts()
itemPurchasePopularity = purchases["itemId"].value_counts()

testset = set(test_queries["queryId"].values)

itemMeanRank = query_item[~query_item["queryId"].isin(testset)][["rank","itemId"]].groupby("itemId")["rank"].mean()
itemMeanRank.name = "itemMeanRank"
itemMaxRank = query_item[~query_item["queryId"].isin(testset)][["rank","itemId"]].groupby("itemId")["rank"].max()
itemMaxRank.name = "itemMaxRank"
itemMinRank = query_item[~query_item["queryId"].isin(testset)][["rank","itemId"]].groupby("itemId")["rank"].min()
itemMinRank.name = "itemMinRank"
itemMedianRank = query_item[~query_item["queryId"].isin(testset)][["rank","itemId"]].groupby("itemId")["rank"].median()
itemMedianRank.name = "itemMedianRank"

itemFeatures = pd.merge(itemMedianRank.reset_index(), itemMaxRank.reset_index()).merge(itemMeanRank.reset_index()).merge(itemMinRank.reset_index())

userView = query_item[["itemId","userView"]].groupby("itemId")["userView"].unique().apply(lambda x:len(x))
userView.name = "uUserView"
userPurchase = query_item[["itemId","userPurchase"]].groupby("itemId")["userPurchase"].unique().apply(lambda x:len(x))
userPurchase.name = "uUserPurchase"

query_item = pd.merge(query_item, userView.reset_index(), how="left")
query_item = pd.merge(query_item, userPurchase.reset_index(), how="left")

queries_str = all_queries[["queryId", "searchstring.tokens"]]
query_item = pd.merge(query_item, queries_str, how="left")

itemViewPopularity.name = "itemView"
itemClickPopularity.name = "itemClick"
itemPurchasePopularity.name = "itemPurchase"

itemFeatures = pd.merge(itemFeatures, itemViewPopularity.reset_index(), left_on="itemId", right_on="index", how="right")
itemFeatures = pd.merge(itemFeatures, itemClickPopularity.reset_index())
itemFeatures = pd.merge(itemFeatures, itemPurchasePopularity.reset_index())

del itemFeatures["index"]

uniqueUserView = item_views[["itemId","userId"]].groupby("itemId")["userId"].unique().apply(lambda x: len(x))
uniqueUserView.name = "uniqueUserView"
itemFeatures = pd.merge(itemFeatures, uniqueUserView.reset_index())

products_info["searchstring.tokens"].fillna("", inplace=True)
products_info["jaccard"] = products_info[["searchstring.tokens","product.name.tokens"]]\
    .apply(lambda x: jaccard(set(x[0].split(",")), set(x[1].split(","))), axis=1)

productNameSize = products_info["product.name.tokens"].apply(lambda x : len(x.split(",")))
productNameSize.name = "productNameSize"  # not using, as it decreased the LB score
products_info = pd.concat((products_info, productNameSize), axis=1)

max_price = products_info[["queryId","pricelog2"]].groupby("queryId")["pricelog2"].max()
min_price = products_info[["queryId","pricelog2"]].groupby("queryId")["pricelog2"].min()
median_price = products_info[["queryId","pricelog2"]].groupby("queryId")["pricelog2"].median()
mean_price = products_info[["queryId","pricelog2"]].groupby("queryId")["pricelog2"].mean()
range_price = max_price - min_price
prices = pd.concat((max_price, min_price, mean_price, median_price, range_price), axis=1)
prices.columns = [["maxp", "minp", "meanp", "medianp", "rangep"]]
products_info = pd.merge(prices.reset_index(), products_info, on="queryId")

products_info["howexpensive"] = 1.0 - (products_info["pricelog2"] / products_info["maxp"])

query_category = pd.merge(query_item[["queryId","itemId"]], products_category)
query_category.sort_values(["queryId","categoryId"], inplace=True)

category_counts = query_category.groupby(["queryId","categoryId"]).agg('count').reset_index().rename(columns = {"itemId":"counts"})
category_counts["catperc"] = category_counts.groupby("queryId")["counts"].apply(lambda x: x / float(x.sum()))
query_category = pd.merge(query_category, category_counts[["queryId","categoryId","catperc"]])

mmr_view = query_item[["rank","itemId"]].where(query_item["viewed"], np.nan, axis=0)
mmr_view = mmr_view.groupby("itemId")["rank"].mean().reset_index().rename(columns={"rank":"mmr_view"})

queriesPerSession = all_queries[["sessionId","queryId"]].groupby("sessionId")["queryId"].count()
queriesPerSession.name = "queriesPerSession"
all_queries = pd.merge(all_queries, queriesPerSession.reset_index())

ndcgPerSession = all_queries[["sessionId","qndcg"]].groupby("sessionId")["qndcg"].mean()
ndcgPerSession.name = "ndcgPerSession"
ndcgPerSession.fillna(ndcgPerSession.mean()/2., inplace=True)
all_queries = pd.merge(all_queries, ndcgPerSession.reset_index())

durationPerSession = all_queries[["sessionId","duration"]].groupby("sessionId")["duration"].mean()
durationPerSession.name = "durationPerSession"
all_queries = pd.merge(all_queries, durationPerSession.reset_index())

querySize = all_queries["searchstring.tokens"].apply(lambda x: 0 if type(x) == np.float else len(x.split(",")))
querySize.name = "querySize"
all_queries = pd.concat((querySize, all_queries), axis=1)

itemFeatures[["itemView", "itemClick", "itemPurchase", "uniqueUserView"]] \
    = minmaxscale(itemFeatures[["itemView", "itemClick", "itemPurchase", "uniqueUserView"]])

query_item[["rank_size", "uUserView", "uUserPurchase"]] = minmaxscale(query_item[["rank_size",
                                                                                  "uUserView", "uUserPurchase"]])

products_info[["pricelog2", "maxp", "minp", "meanp", "medianp", "rangep", "productNameSize"]] \
    = minmaxscale(products_info[["pricelog2", "maxp", "minp", "meanp", "medianp", "rangep", "productNameSize"]])

result = pd.merge(query_item[["queryId","itemId","clicked","viewed","purchased","rank",
                              "rank_size","uUserView","uUserPurchase"]], itemFeatures, how="left")

result = pd.merge(result, products_info[["queryId","itemId","pricelog2","jaccard","maxp","minp",
                                         "meanp","medianp","rangep","howexpensive","productNameSize"]],
                  on=["queryId", "itemId"], how="left")

result = pd.merge(result, query_category[["queryId","itemId","catperc"]], how="left")

terms_item_events = pd.merge(all_queries[["queryId","searchstring.tokens"]], query_item[["queryId","itemId","clickTime","viewTime","purchaseTime"]], on="queryId")
tie = terms_item_events.groupby(["searchstring.tokens","itemId"])[["clickTime","viewTime","purchaseTime"]].count()
tie.columns = ["count_clicks","count_views","count_purchases"]

event_counts = pd.merge( tie.reset_index(), terms_item_events[["itemId","queryId","searchstring.tokens"]])
event_counts[["count_clicks", "count_views", "count_purchases"]] = minmaxscale(event_counts[["count_clicks","count_views","count_purchases"]])

#
result = pd.merge( result, event_counts[["itemId","queryId","count_clicks","count_views","count_purchases"]])
#

all_queries[["querySize", "avg_duration", "queriesPerSession", "ndcgPerSession", "durationPerSession"]] \
    = minmaxscale(all_queries[["querySize", "avg_duration", "queriesPerSession", "ndcgPerSession",
                           "durationPerSession"]])

result = pd.merge(result, all_queries[["queryId", "avg_ndcg", "avg_duration", "querySize", "queriesPerSession",
                                   "ndcgPerSession", "durationPerSession"]])

result = pd.merge(result, mmr_view, how="left")

result["mmr_view"] = result["mmr_view"].fillna(result["mmr_view"].mean()/.2)
result["itemMedianRank"] = result["itemMedianRank"].fillna(result["itemMedianRank"].mean()/2.)
result["itemMaxRank"] = result["itemMaxRank"].fillna(result["itemMaxRank"].mean()/2.)
result["itemMinRank"] = result["itemMinRank"].fillna(result["itemMinRank"].mean()/2.)
result["itemMeanRank"] = result["itemMeanRank"].fillna(result["itemMeanRank"].mean()/2.)

result.fillna(0.0, inplace=True)
result.sort_values(["queryId","itemId"], inplace=True)

create_output(result, train_queries, test_queries)

