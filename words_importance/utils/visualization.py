import numpy as np
import pandas as pd

import altair as alt

import seaborn as sns
import matplotlib.pyplot as plt


# plot pairwise scatter plots of bert embeddings
def pairwise_bertembeddings(
    bert_embeddings,
    pairlist,
    nrows,
    ncols,
    savepath=None,
    fake_embeddings=None,
    kde=False,
):
    p = len(bert_embeddings[0])
    if pairlist is None:
        pairlist = [np.random.choice(range(p), 2, False) for _ in range(nrows * ncols)]

    assert (
        len(pairlist) == nrows * ncols
    ), "Length of pairlist does not mathc specivied nrows and ncols"

    if fake_embeddings is not None:
        assert len(bert_embeddings[0]) == len(
            fake_embeddings[0]
        ), "Dimension mismatch between bert_embeddings and fake_embeddings"

    fig = plt.figure(figsize=(12 * ncols, 10 * nrows))
    for i in range(nrows * ncols):
        c1, c2 = pairlist[i]
        plt.subplot(nrows, ncols, 1 + i)
        plt.title(f"x_{c1 + 1} Versus x_{c2 + 1}", fontsize=20)
        plt.xlabel(f"x_{c1 + 1}", fontsize=16)
        plt.tick_params(axis="x", labelsize=16)
        plt.tick_params(axis="y", labelsize=16)
        plt.ylabel(f"x_{c2 + 1}", fontsize=16)
        if kde:
            sns.kdeplot(
                x=bert_embeddings[:, c1],
                y=bert_embeddings[:, c2],
                cmap="Blues",
                shade=True,
            )
            plt.scatter([], [], s=20, alpha=0.3, label="True", color="C0")
        else:
            plt.scatter(
                bert_embeddings[:, c1],
                bert_embeddings[:, c2],
                s=20,
                alpha=0.3,
                label="True",
            )

        if fake_embeddings is not None:
            plt.scatter(
                fake_embeddings[:, c1],
                fake_embeddings[:, c2],
                s=20,
                alpha=0.3,
                color="firebrick",
                label="Flow",
            )

            plt.legend(fontsize=18)
    if savepath is not None:
        plt.savefig(savepath)
    plt.close()
    return fig


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
