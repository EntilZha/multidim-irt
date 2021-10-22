library(ggplot2)
library(readr)
library(dplyr)
library(gridExtra)
library(jsonlite)
library(stringr)
library(pROC)
# file

datasets <- c(
    "anli-r1-dev.jsonl.preds.jsonlines",
    "anli-r2-dev.jsonl.preds.jsonlines",
    "anli-r3-dev.jsonl.preds.jsonlines",
    "superglue-winogender.jsonl.preds.jsonlines"
    )

buildPlot  <- function(dataset){
#    D.amortized <- jsonlite::read_json(str_glue("../py-irt/results_amortized_{dataset}/best_parameters.json"),
#                                       simplify=TRUE)

#    D.1pl <- jsonlite::read_json(str_glue("../py-irt/results_mean_field_{dataset}/best_parameters.json"),
#                                 simplify=TRUE)

    D.amortized <- stream_in(
        file(
            str_glue(
                "../py-irt/results/amortized/nli/{dataset}/model_predictions.jsonlines"
            )
        )
    )
    

    D.1pl <- stream_in(
        file(
            str_glue(
                "../py-irt/results/meanfield/nli/{dataset}/model_predictions.jsonlines"
            )
        )
    )
    
    D.1pl$type  <- "meanfield"
    D.amortized$type  <- "amortized"

    auc.1pl  <- round(auc(D.1pl$response, D.1pl$prediction), 3)
    auc.amortized  <- round(auc(D.amortized$response, D.amortized$prediction), 3)
    dname  <- strsplit(dataset, "\\.")[[1]][[1]]

    plot_title  <- str_glue(
        "{dname} -- Amortized AUC: {auc.amortized}, Meanfield AUC: {auc.1pl}"
    )

    D  <-  D.amortized %>%
        bind_rows(D.1pl)
    p  <- ggplot(aes(x=prediction, y=..scaled.., color=type, fill=type), data=D) +
        geom_density(fill=NA) +
        ggtitle(plot_title)
    return(p)
} 


plots  <- lapply(datasets, buildPlot)

p 

do.call(grid.arrange, plots)
