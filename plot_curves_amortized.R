library(ggplot2)
library(readr)
library(dplyr)
library(gridExtra)
library(jsonlite)
library(stringr)
# file

datasets <- c(
    "aqa-r1",
    "bio-asq",
    "drop_dev",
    "duo_rc",
    "news_qa",
    "race_dev",
    "relation_extraction"
    )

buildPlot  <- function(dataset){
#    D.amortized <- jsonlite::read_json(str_glue("../py-irt/results_amortized_{dataset}/best_parameters.json"),
#                                       simplify=TRUE)

#    D.1pl <- jsonlite::read_json(str_glue("../py-irt/results_mean_field_{dataset}/best_parameters.json"),
#                                 simplify=TRUE)

    D.amortized <- stream_in(
        file(
            str <- glue(
                "../py-irt/results_mean_field_{dataset}/model_predictions.jsonlines"
            )
        )
    )
    

    D.1pl <- stream_in(
        file(
            str <- glue(
                "../py-irt/results_mean_field_{dataset}/model_predictions.jsonlines"
            )
        )
    )
    
    D.1pl$type  <- "meanfield"
    D.amortized$type  <- "amortized"

    D  <-  D.amortized %>%
        bind_rows(D.1pl)
    p  <- ggplot(aes(x=difficulty, y=..scaled.., color=type, fill=type), data=D) +
        geom_density(fill=NA) +
        ggtitle(dataset)
    return(p)
}


plots  <- lapply(datasets, buildPlot)

p 

do.call(grid.arrange, plots)
