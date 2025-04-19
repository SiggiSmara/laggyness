# set up some path constants
from pathlib import Path

my_path = Path(__file__).parent.parent
data_path = my_path / "data"
combined_path = data_path / "stocks_1d_combined"
trainsets_path = data_path / "stocks_1d_trainsets"

trainsets_path.mkdir(exist_ok=True)

# get list of all tickers
tickers = [ x for x in (combined_path).glob("*.parquet")]
