import gzip
import numpy as np
import polars as pl
import numpy as np
import hftbacktest as hbt
from hftbacktest.data.utils import binancefutures
import numpy as np

from numba import njit, uint64
from numba.typed import Dict

from hftbacktest import (
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    GTX,
    LIMIT,
    BUY,
    SELL,
    BUY_EVENT,
    SELL_EVENT,
    Recorder,
)
from hftbacktest.stats import LinearAssetRecord

from hftbacktest import LOCAL_EVENT, EXCH_EVENT

@njit
def generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp):
    for i in range(len(data)):
        exch_ts = data[i].exch_ts
        local_ts = data[i].local_ts
        feed_latency = local_ts - exch_ts
        order_entry_latency = mul_entry * feed_latency + offset_entry
        order_resp_latency = mul_resp * feed_latency + offset_resp

        req_ts = local_ts
        order_exch_ts = req_ts + order_entry_latency
        resp_ts = order_exch_ts + order_resp_latency

        order_latency[i].req_ts = req_ts
        order_latency[i].exch_ts = order_exch_ts
        order_latency[i].resp_ts = resp_ts

def generate_order_latency(feed_file, output_file = None, mul_entry = 1, offset_entry = 0, mul_resp = 1, offset_resp = 0):
    data = np.load(feed_file)['data']
    df = pl.DataFrame(data)

    df = df.filter(
        (pl.col('ev') & EXCH_EVENT == EXCH_EVENT) & (pl.col('ev') & LOCAL_EVENT == LOCAL_EVENT)
    ).with_columns(
        pl.col('local_ts').alias('ts')
    ).group_by_dynamic(
        'ts', every='1000000000i'
    ).agg(
        pl.col('exch_ts').last(),
        pl.col('local_ts').last()
    ).drop('ts')

    data = df.to_numpy(structured=True)

    order_latency = np.zeros(len(data), dtype=[('req_ts', 'i8'), ('exch_ts', 'i8'), ('resp_ts', 'i8'), ('_padding', 'i8')])
    generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp)

    if output_file is not None:
        np.savez_compressed(output_file, data=order_latency)

    return order_latency



def binance_normalize(file_name):
    file_output = 'data/'+file_name+'.npz'
    file_input="data/"+file_name+".gz"
    
    data = binancefutures.convert(
    file_input,
    combined_stream=True,
    output_filename=file_output,

)
    return pl.DataFrame(data)

def read_binance_gz(file_name):
    file_input="data/"+file_name+".gz"
    try:
        data= binance_normalize(file_name)
    except:
        content = []
        with gzip.open(file_input, 'rt') as f:
            for line_number, line in enumerate(f, start=1):
                content.append(line)

        final_content = ''.join(content[:10])
        with gzip.open(file_input, 'wt') as f:
            f.write(final_content)
        data= binance_normalize(file_name)
    order_latency = generate_order_latency("data/"+file_name+'.npz', output_file='data/'+file_name+'_latency.npz', mul_entry=4, mul_resp=3)

    return data
read_binance_gz('1000bonkusdt_20240730')

