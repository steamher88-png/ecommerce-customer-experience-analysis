import pandas as pd
import duckdb 
import matplotlib.pyplot as plt
import numpy as num
import statsmodels.formula.api as stat

file2 = r'C:\Users\steam\OneDrive\Desktop\Key\ecommerce_sales_customer_analytics_150k.csv'
file3 = r'C:\Users\steam\OneDrive\Desktop\Key\order_items.csv'
file4 = r'C:\Users\steam\OneDrive\Desktop\Key\product_catalog.csv'

ecommerce_sales_customer = pd.read_csv(file2, sep = ',')
order_items = pd.read_csv(file3, sep = ',')
product_catalog = pd.read_csv(file4, sep = ',')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)



duckdb.query('''
CREATE OR REPLACE TABLE clear AS
select prod.product_id, prod.product_category, max(prod.product_rating) as rating, count(*) as count_orders
from ecommerce_sales_customer as ecom
LEFT JOIN order_items AS ord ON ecom.order_id = ord.order_id
LEFT JOIN product_catalog AS prod ON prod.product_id = ord.product_id

where ecom.order_status = 'Completed' 
group by prod.product_id,prod.product_category
''')

duckdb.query('''
CREATE OR REPLACE TABLE order_od AS
WITH items AS (
   SELECT order_id, 
   SUM(net_sales) AS items_net_sales,
   CASE 
     WHEN ROUND(SUM(quantity * unit_price + tax_amount + shipping_cost- discount_amount), 2) = ROUND(SUM(net_sales), 2) 
     THEN 1 
     ELSE 0 
     END AS items_is 
   FROM order_items 
   GROUP BY order_id
),

prod AS (
   SELECT order_id, AVG(prod.product_rating) AS prod_rating
   FROM order_items
   JOIN product_catalog AS prod ON prod.product_id = order_items.product_id
   GROUP BY order_items.order_id
)

SELECT ecom.*,
       ecom.delivery_days - ecom.estimated_delivery_days AS delay_days,
       prod.prod_rating,
       COUNT(CASE WHEN items.items_is = 1 THEN 1 END) OVER () AS matching_rows
FROM ecommerce_sales_customer AS ecom
JOIN items ON ecom.order_id = items.order_id
LEFT JOIN prod ON prod.order_id = ecom.order_id
WHERE ecom.customer_rating IS NOT NULL 

''')

duckdb.query('''
CREATE OR REPLACE TABLE table_order AS
   SELECT order_items.order_id, prod.product_rating AS prod_rating
   FROM order_items
   JOIN product_catalog AS prod ON prod.product_id = order_items.product_id
    
    ''')

clear = duckdb.query('SELECT * FROM clear').df()
order = duckdb.query('SELECT * FROM order_od').df()
table = duckdb.query('SELECT * FROM table_order').df()



#Korell clear (product - rating - relations) 
#print(clear)
#print(clear['count_orders'].describe())

print('\n')

print('Korrel')
print(clear['rating'].corr(clear['count_orders']))
print(clear['rating'].corr(clear['count_orders'], method='spearman'))

print('\n')
print('Group kor')
print('\n')

for cat, pat in clear.groupby('product_category'):
    n = len(pat)
    kor = pat['rating'].corr(pat['count_orders'])
    border = 2 / (n**0.5)
    print(cat, n, 'Korrel:', round(kor, 3), 'Border:', round(border, 3))

print('\n')

print('Numpy')
n = len(clear)
print('Product:', n)
print('SR:', clear['count_orders'].mean())
print('SD:', clear['count_orders'].std())
print('Razbros:', num.sqrt(clear['count_orders'].mean()))

print('\n')

print('Other')
plt.scatter(clear['rating'],clear['count_orders'] )
plt.show()

#print(order)

#order_test = duckdb.query('SELECT count(distinct(order_id)) FROM order_od').df()
#order_test1 = duckdb.query('SELECT count(*) FROM order_od').df()
#print(order_test,order_test1 )

#Proverka korrelacii (Total)
for metric in order.select_dtypes(include='number').columns:
    if metric in ['customer_rating', 'items_is', 'matching_rows']:
        continue
    part = order[[metric, 'customer_rating']]
    n = len(part)
    kor = part[metric].corr(part['customer_rating'])
    korsper = part[metric].corr(part['customer_rating'],method = 'spearman')
    border = 2 / (n ** 0.5)
    print(metric, n,
          'Korrel Pirs:', round(kor, 3),
          'Korrel Spear:', round(korsper,3),
          'Determin', round(kor ** 2, 4),
          'Border:', round(border, 3),
          'Abroad:', abs(kor) > border)

#Graf net_sales - customer_rating
plt.scatter(order['net_sales'],order['customer_rating'],s=10,alpha=0.05)
plt.show()


#Delivery
print(order.groupby(['delay_days'])['customer_rating'].agg(['mean','count']))
print('\n')

#status
print('Table delivery_status')
print( order.groupby('delivery_status')['customer_rating'].agg(['mean', 'std', 'count'])) 
print('\n')

#customer
print(order.groupby(['customer_segment', 'delivery_status'])['customer_rating'].agg(['mean', 'std', 'count']))
print('\n')

order['order_date'] = pd.to_datetime(order['order_date'])
order['year'] = order['order_date'].dt.year
order['month'] = order['order_date'].dt.month
order['group_age'] = pd.cut(order['customer_age'], [0,15,25,35,45,55,65,75,85,95,120])

columns = ['sales_channel', 'customer_country', 'shipping_method',
          'quantity', 'group_age', 'year', 'month']

for col in columns:
    result = order.groupby(col,observed=True)['customer_rating'].agg(['mean','std', 'count'])
    margin = 1.96 * result['std'] / (result['count'] ** 0.5)
    result['ci_low'] = result['mean'] - margin
    result['ci_high'] = result['mean'] + margin
    print('\n', col)
    print(result.round(3))
print('\n')

#product_rating - customer_rating
print(order['prod_rating'].isna().sum()) # 0
print(order['prod_rating'].corr(order['customer_rating']))
print(order['prod_rating'].corr(order['customer_rating'], method='spearman'))
order['prod_rating_group'] = pd.cut(order['prod_rating'], [1, 2, 3, 3.5, 4, 4.5, 5 ])
print('\n')
print(
    order.groupby('prod_rating_group', observed=True)['customer_rating'].agg(['count', 'mean', 'std'])
)
#product_rating - customer_rating//regress///model
model = stat.ols(
    "customer_rating ~ C(delivery_status, Treatment('On Time')) + prod_rating"
    " + quantity + C(customer_segment, Treatment('Consumer'))", data=order
).fit()

print(model.summary().tables[1])
print('Determin:', round(model.rsquared, 3))
print('\n')

#customer behavior
print(order.groupby('customer_id')['customer_order_count'].nunique().max())  
cut = order.groupby('customer_id').agg( 
    avg_rating = ('customer_rating', 'mean'),
    con_orders = ('customer_order_count', 'first'),
    repeat = ('is_repeat_customer', 'first'),
)
print('Pirs(avg_rating):', cut['avg_rating'].corr(cut['con_orders']))
print('Spear(avg_rating):', cut['avg_rating'].corr(cut['con_orders'],method='spearman'))
print(cut.groupby('repeat')['avg_rating'].agg(['mean', 'count']))
print('\n')

#model
b = model.params
k = "C(delivery_status, Treatment('On Time'))[T.Delayed]"

share_late = (order['delivery_status'] == 'Delayed').mean()
print('\nDelayed round:', round(share_late, 4))
print('- 5 p.p.:', round(0.05 * -b[k], 3))
print('Delete:', round(share_late * -b[k], 3))
print('\n')

for th in [3.0, 3.5, 4.0]:
    sidor2 = table.copy()
    sidor2['prod_rating'] = sidor2['prod_rating'].clip(lower=th)
    new_rating = sidor2.groupby('order_id')['prod_rating'].mean()
    order2 = order.copy()
    order2['prod_rating'] = order2['order_id'].map(new_rating)
    total = (model.predict(order2) - model.predict(order)).mean()
    changed_ids = table.loc[table['prod_rating'] < th, 'order_id']
    touched = order['order_id'].isin(changed_ids).mean()
    print(th,'Total:', round(total, 3),'Touched %:', round(touched * 100, 1))


#Tables
print('ecommerce_sales_customer')
print(ecommerce_sales_customer.head())
print(ecommerce_sales_customer.dtypes)
print(ecommerce_sales_customer.shape)
print(ecommerce_sales_customer.isna().mean().sort_values(ascending=False))
print(ecommerce_sales_customer.describe(include='all'))
print(ecommerce_sales_customer.duplicated().sum())

print('____________________________')
print('order_items')
print(order_items.head())
print(order_items.dtypes)
print(order_items.shape)
print(order_items.isna().mean().sort_values(ascending=False))
print(order_items.describe(include='all'))
print(order_items.duplicated().sum())

print('____________________________')
print('product_catalog')
print(product_catalog.head())
print(product_catalog.dtypes)
print(product_catalog.shape)
print(product_catalog.isna().mean().sort_values(ascending=False))
print(product_catalog.describe(include='all'))
print(product_catalog.duplicated().sum())

print('\n')
print('Other EDA')
print(ecommerce_sales_customer.isna().mean().sort_values(ascending=False))
print(ecommerce_sales_customer['customer_rating'].isna().groupby(ecommerce_sales_customer['order_status']).mean())
print(pd.crosstab(order['delivery_status'], order['delay_days']))
