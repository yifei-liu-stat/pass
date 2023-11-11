import pandas as pd

import altair as alt


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    """convert a dense matrix to a data frame with row and column indices"""
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (
                    r,
                    row_tokens[r] if len(row_tokens) > r else "<blank>",
                ),  # same as f-string
                "%.3d %s"
                % (
                    c,
                    col_tokens[c] if len(col_tokens) > c else "<blank>",
                ),  # same as f-string
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn_mat, row_tokens, col_tokens, max_dim=30):
    """Visualize attention matrix"""
    df = mtx2df(
        attn_mat,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=500, width=500)
        .interactive()
    )
