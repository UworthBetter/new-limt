"""Explicit run dates and dated listing eligibility; source arrays stay intact."""
from datetime import date
import numpy as np

DEFAULT_EVALUATION_END = '2026-06-30'
LEGACY_EVALUATION_END = '2026-09-30'
DELISTINGS = {
    'SH600837': dict(date='2025-03-04', name='海通证券',
        source='https://www.hkexnews.hk/listedco/listconews/sehk/2025/0225/2025022501027_c.pdf'),
    'SH601989': dict(date='2025-09-05', name='中国重工',
        source='https://static.cninfo.com.cn/finalpage/2025-08-30/1224625027.PDF'),
}


def splits_for(evaluation_end=DEFAULT_EVALUATION_END):
    if not isinstance(evaluation_end, str):
        raise ValueError('evaluation_end must be an ISO date YYYY-MM-DD')
    try:
        parsed=date.fromisoformat(evaluation_end)
    except ValueError as error:
        raise ValueError('evaluation_end must be an ISO date YYYY-MM-DD') from error
    if parsed.isoformat()!=evaluation_end or evaluation_end < '2023-01-01':
        raise ValueError('evaluation_end must be an ISO date on/after 2023-01-01')
    return {'train':('2008-01-01','2018-12-31'), 'valid':('2019-01-01','2020-12-31'),
        'rule_select':('2021-01-01','2022-12-31'), 'eval':('2023-01-01',evaluation_end)}


def listing_eligible(dates, symbols):
    dates=np.asarray(dates,dtype='datetime64[D]')
    symbols=np.asarray(symbols).astype(str)
    eligible=np.ones((len(dates),len(symbols)),bool)
    for symbol,rule in DELISTINGS.items():
        columns=symbols==symbol
        eligible[:,columns] &= (dates<np.datetime64(rule['date']))[:,None]
    return eligible


def signal_days(dates, split, evaluation_end=DEFAULT_EVALUATION_END, lookback=1, training=False, limit=0):
    dates=np.asarray(dates,dtype='datetime64[D]')
    start,end=map(np.datetime64,splits_for(evaluation_end)[split])
    last=min(int(np.searchsorted(dates,end,side='right'))-1,len(dates)-1)
    # Final accounting mark is t+2; the longest supervised target ends at t+6.
    days=np.arange(max(lookback-1,int(np.searchsorted(dates,start))),last-(6 if training else 2)+1)
    if limit:days=days[:limit]
    if len(days)<2:raise ValueError(f'empty/too short {split} domain')
    return days
