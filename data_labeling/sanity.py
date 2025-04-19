import polars as pl
from datetime import date
import altair as alt
from rich.progress import track
from pathlib import Path

from common import (
    find_data_all_lazy,
    data_path,
    centrally_smoothed_path,
    windows,
    intermediate_path,
)
# check_path = centrally_smoothed_path / "AMN.parquet"

images_path = intermediate_path / "labeling" / "images"

# find the eligible tickers
all_tickers = list(centrally_smoothed_path.glob("*.parquet"))
# subset = all_tickers[list(range(0,len(all_tickers), 100))]
graph_ending = "html"
ch_width = 1600
ch_height = 600
mark_point_size = 50

trend_windows = [5, 7, 9]

brush = alt.selection_interval(encodings=['x'])

for i in track(range(0,len(all_tickers), 1000), description="Plotting result..."):
    check_path = Path(all_tickers[i])
    q = pl.read_parquet(check_path).drop_nulls() #.filter((pl.col("date") > date(year=2020, month=1, day=1)) & (pl.col("date") < date(year=2022, month=1, day=1)) ) #
    
    orig = alt.Chart(q).mark_line().encode(
        x='date',
        y=alt.Y('original'),
        color=alt.value("#999999")
    ).properties(width=ch_width, height=ch_height)
    orig.interactive().save(images_path / f"{check_path.stem}_original.{graph_ending}")

    


    # print(q.columns)
    for window in windows:
        orig_name = f"savgol_p2_{window}"
        diff_name = f"{orig_name}_diff"
        rdiff_name = f"{orig_name}_rel_diff"
        # label_name = f"{rdiff_name}_label"
        # trend9_name = f"{rdiff_name}_9_trend"
        # trend7_name = f"{rdiff_name}_7_trend"
        # trend5_name = f"{rdiff_name}_5_trend"
        # label9_name = f"{rdiff_name}_label9"
        # label7_name = f"{rdiff_name}_label7"
        # label5_name = f"{rdiff_name}_label5"
        # small_name = f"{rdiff_name}_is_small"
        # pos_name = f"{rdiff_name}_is_positive"
        # neg_name = f"{rdiff_name}_is_negative"
        # posneg_name = f"{rdiff_name}_posneg"
        # seven_trend_name = f"{rdiff_name}_7_trend"
        # five_trend_name = f"{rdiff_name}_5_trend"

        smooth = alt.Chart(q).mark_line().encode(
            x='date',
            y=alt.Y(orig_name),
            color=alt.value("#666666")
        ).properties(width=ch_width, height=ch_height)
        (orig + smooth).interactive().save(images_path / f"{check_path.stem}_{orig_name}.{graph_ending}")

        upper = (orig + smooth).encode(alt.X('date').scale(domain=brush))
        # qpl = q.plot.line(x="date", y=orig_name).properties(width=ch_width, height=ch_height)
        


        # qpl = q.plot.line(x="date", y=rdiff_name).properties(width=ch_width, height=ch_height/3).add_params(brush)
        # alt.vconcat(upper, qpl).save(images_path / f"{check_path.stem}_{rdiff_name}.{graph_ending}")

        for tw in trend_windows:
            trend_name = f"{rdiff_name}_{tw}_trend"
            label_name = f"{rdiff_name}_label{tw}"
            index_name = f"{rdiff_name}_labelindex{tw}"
            # print(q.select([label_name, index_name]).head(10))
            
            
            qpl = alt.Chart(q).mark_line().encode(
                x='date', y=alt.Y(label_name),
            ).properties(
                width=ch_width, height=ch_height/3
            ).add_params(brush)
            # q.plot.line(x="date", y=label_name).properties(width=ch_width, height=ch_height/3).add_params(brush)
            alt.vconcat(upper, qpl).save(images_path / f"{check_path.stem}_{label_name}.{graph_ending}")

            # qpl = q.plot.line(x="date", y=index_name).save(images_path / f"{check_path.stem}_{index_name}.{graph_ending}")

            qpl = alt.Chart(q).mark_point(size=mark_point_size).encode(
                x='date',
                y=alt.Y('original'),
                color=alt.Color(
                    f"{label_name}:N", 
                    scale=alt.Scale(
                        domain=[0, 0.5, 1], 
                        range=['red','blue','green'], 
                        # interpolate=method
                    ),
                )
            ).properties(width=ch_width, height=ch_height)
            (orig + smooth + qpl).interactive().save(images_path / f"{check_path.stem}_original_labelled{tw}_{window}.{graph_ending}", scale_factor=3)

        # qpl = q.plot.line(x="date", y=label9_name).properties(width=ch_width, height=ch_height).interactive()
        # (orig + smooth + qpl).save(images_path / f"{check_path.stem}_{label9_name}.{graph_ending}", scale_factor=3)
        # qpl = q.plot.line(x="date", y=label7_name).properties(width=ch_width, height=ch_height).interactive()
        # (orig + smooth + qpl).save(images_path / f"{check_path.stem}_{label7_name}.{graph_ending}", scale_factor=3)
        # qpl = q.plot.line(x="date", y=label5_name).properties(width=ch_width, height=ch_height).interactive() 
        # (orig + smooth + qpl).save(images_path / f"{check_path.stem}_{label5_name}.{graph_ending}", scale_factor=3)

        # qpl = alt.Chart(q).mark_point(size=mark_point_size).encode(
        #     x='date',
        #     y=alt.Y('original'),
        #     color=alt.Color(
        #         f"{label7_name}:N", 
        #         scale=alt.Scale(
        #             domain=[-1, 0, 1], 
        #             range=['red','blue','green'], 
        #             # interpolate=method
        #         ),
        #     )
        # ).properties(width=ch_width, height=ch_height).interactive() 
        # (orig + smooth + qpl).save(images_path / f"{check_path.stem}_original_labelled7_{window}.{graph_ending}", scale_factor=3)

        # qpl = alt.Chart(q).mark_point(size=mark_point_size).encode(
        #     x='date',
        #     y=alt.Y('original'),
        #     color=alt.Color(
        #         f"{label5_name}:N", 
        #         scale=alt.Scale(
        #             domain=[-1, 0, 1], 
        #             range=['red','blue','green'], 
        #             # interpolate=method
        #         ),
        #     )
        # ).properties(width=ch_width, height=ch_height).interactive() 
        # (orig + smooth + qpl).save(images_path / f"{check_path.stem}_original_labelled5_{window}.{graph_ending}", scale_factor=3)


    


# .with_row_index() 
# filter on label.shift(1) != label.current 
# if end - start > x (5? 10?) then real trend 
# if one or two pointa between two same trends then grow together 
# anything else label as unknown 