library(ggplot2)
library(readr)
library(dplyr)
library(gridExtra)
library(jsonlite)
library(stringr)
library(pROC)
# file

data.nli <- c(
    "nli/anli-r1-dev.jsonl.preds.jsonlines",
    "nli/anli-r2-dev.jsonl.preds.jsonlines",
    "nli/anli-r3-dev.jsonl.preds.jsonlines",
    "nli/superglue-winogender.jsonl.preds.jsonlines"
)

data.sentiment  <- c(
    "sentiment/dynasent-r1-dev.jsonl.preds.jsonlines",
    "sentiment/dynasent-r2-dev.jsonl.preds.jsonlines",
    "sentiment/sst3-dev.jsonl.preds.jsonlines"
)

data.hs <- c(
    "hs/ahs-r1-dev.jsonl.preds.jsonlines",
    "hs/ahs-r2-dev.jsonl.preds.jsonlines",
    "hs/ahs-r3-dev.jsonl.preds.jsonlines",
    "hs/ahs-r4-dev.jsonl.preds.jsonlines"
)

buildPlot  <- function(dataset){
#    D.amortized <- jsonlite::read_json(str_glue("../py-irt/results_amortized_{dataset}/best_parameters.json"),
#                                       simplify=TRUE)

#    D.1pl <- jsonlite::read_json(str_glue("../py-irt/results_mean_field_{dataset}/best_parameters.json"),
#                                 simplify=TRUE)

    D.amortized <- stream_in(
        file(
            str_glue(
                "../py-irt/results/amortized/{dataset}/model_predictions.jsonlines"
            )
        )
    )
    

    D.1pl <- stream_in(
        file(
            str_glue(
                "../py-irt/results/meanfield/{dataset}/model_predictions.jsonlines"
            )
        )
    )
    
    D.1pl$type  <- "meanfield"
    D.amortized$type  <- "amortized"

    auc.1pl  <- round(auc(D.1pl$response, D.1pl$prediction), 3)
    auc.amortized  <- round(auc(D.amortized$response, D.amortized$prediction), 3)
    dname  <- strsplit(dataset, "\\.")[[1]][[1]]
    dname  <- strsplit(dname, "\\/")[[1]][[2]]

    plot_title  <- str_glue(
        "{dname}"
    )
    plot_text <- str_glue(
        "Am. AUC: {auc.amortized} MF AUC: {auc.1pl}"
    )

    D  <-  D.amortized %>%
        bind_rows(D.1pl)
    p  <- ggplot(aes(x=prediction, y=..scaled.., color=type, fill=type), data=D) +
        geom_density(fill=NA) +
        ggtitle(plot_title) +
        annotate("text", x=0.5, y=0, label=plot_text)
    return(p)
} 



plots.nli  <- lapply(data.nli, buildPlot)

p <- do.call(
    arrangeGrob,
    plots.nli
)

ggsave(
    "nli.png",
    p,
    width=10,
    height=5
)


# sentiment 
plots.sentiment  <- lapply(data.sentiment, buildPlot)

p <- do.call(
    arrangeGrob,
    c(plots.sentiment,nrow=2)
)

ggsave(
    "sentiment.png",
    p,
    width=8,
    height=5
)


# HS
plots.hs  <- lapply(data.hs, buildPlot)

p <- do.call(
    arrangeGrob,
    plots.hs
)

ggsave(
    "hs.png",
    p,
    width=10,
    height=5
)




