#' Gradient-Boosted Trees Classifier using the gbm package with ordinal encoding.
#'
#' @Author: Mark Steadman \email{mark@datarobot.com}
#' @Copyright: 2014 DataRobot Inc
#' @rtype: Binary
#' @row_limit: 60000
#'
#' This program is free software; you can redistribute it and/or
#' modify it under the terms of the GNU General Public License
#' as published by the Free Software Foundation; either version 2
#' of the License, or (at your option) any later version.
#'
#' This program is distributed in the hope that it will be useful,
#' but WITHOUT ANY WARRANTY; without even the implied warranty for
#' MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#' GNU General Public License for more details.
#'
#' You should have received a copy of the GNU General Public License
#' along with this program; if not, write to the Free Software
#' Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

modelfit <- function(response, data, extras) {
  # Fits a model using gbm
  #
  # Args:
  #   response: a vector containing the response values (classification problem)
  #   data: a data frame containing the training data
  #   extras: a list containing any extras to be passsed
  #
  # Returns:
  #   model: a list containing all state information needed to
  #          predict on  new data
  

  # Cardinality threshold for convesion to numeric
  CARD_MAX <- 0

  # Set the dist argument to "bernoulli" for classification 
  DIST <- "bernoulli"

  
  # Number of rounds of boosting to undertake 
  # And the shrinkage applied
    NTREES <- 400
    SHRINK <- 0.05


  # Any categorical variable that occurs fewer than SUPPORT_MIN times will be 
  # grouped together in a category by themselves
  SUPPORT_MIN <- max(5,nrow(data)/1000) 


  # Check to ensure that it is a binary classification problem
  if (sum(!is.na(unique(response))) > 2) {
    stop("Supports binary classification only")
  }

  # remove NA targets
  data <- data[!is.na(response),]
  response <- response[!is.na(response)]


  # Load the gdata package which provides the maplevels functionality
  # to encode factor levels
  library(gdata)

  # Load the gbm package which implements the gradient-boosting machine
  library(gbm)
  
  # Record the time used for preprocessing for now  
  start_time <- proc.time()
  
  # Randomly permute the rows since we might have all of one class at the end
  # and to ensure that the validation set on the number of trees to use
  # is representative

  # Set the random seed
  set.seed(127127)

  new_rows <- sample(length(response))
  response <- response[new_rows]
  data <- data[new_rows, ]
  
  # Categorical variable preprocessing

  # Separate out character variables and encode as factor levels
  mchar_cols <- names(which(sapply(data, FUN=is.character)))
  
  # Convert to factor levels
  data[mchar_cols] <- lapply(data[mchar_cols], FUN=factor)

  # Keep cols if not all the same and not all different
  col_levels <- lapply(data[mchar_cols], FUN=nlevels)
  char_cols_used <- names(which((col_levels > 1) & (col_levels < (0.95 * nrow(data)))))
  
  # Create empty vectors and lists in case we skip the main processing loop  
  mchar_cols_low_card <-vector()
  mchar_cols_high_card <- vector()
  mmodes <- list()
  mfactor_maps <- list()
  
  # Check that at least one passes
  if (length(char_cols_used > 0)) {
    data[char_cols_used] <- lapply(data[char_cols_used], FUN=addNA)
    
    # Record factor levels
    mfactor_maps <- lapply(data[char_cols_used], FUN=function(X) { return(mapLevels(X, codes=FALSE))})
    
    # Record most common class
    mmodes <- lapply(data[char_cols_used], FUN=function(X) { return(levels(X)[which.max(as.numeric(table(X)))])})
    
    # Create a list of the column names for the high-cardinality categorical variables  
    mchar_cols_high_card <- names(which(col_levels >= CARD_MAX))
    mchar_cols_high_card <- mchar_cols_high_card[which(mchar_cols_high_card %in% char_cols_used)]

    # And another list with any low-cardinality categorical variables
    if(length(mchar_cols_high_card) > 0) {
      mchar_cols_low_card <- setdiff(char_cols_used,  mchar_cols_high_card)
    } else {
      mchar_cols_low_card <- cahr_cols_used 
    }
    
    # Reorder based on frequency for high_cardinality variables
    # Also combine the levels which have low support
    if(length(mchar_cols_high_card) > 0) {
      for(i in 1:length(mchar_cols_high_card)) {
        cur_col <- mchar_cols_high_card[i]
        freq_table <- table(data[mchar_cols][[cur_col]])
        
        low_support <- names(freq_table)[as.numeric(freq_table) < SUPPORT_MIN]
        
        table_ord <- order(as.numeric(freq_table), decreasing=TRUE)
        
        levels(data[mchar_cols][[cur_col]])[levels(data[mchar_cols][[cur_col]]) %in% low_support] <- "--Low Support--"
        data[mchar_cols][[cur_col]] <-  factor(data[mchar_cols][[cur_col]], levels(data[mchar_cols][[cur_col]][table_ord]))
      }
    }
    
    # Record factor levels for later
    mfactor_maps <- lapply(data[char_cols_used], FUN=function(X) { return(mapLevels(X, codes=FALSE))})
    
    # Convert high card into numeric, we leave low-cardinality as factor levels
    data[mchar_cols_high_card] <- lapply(data[mchar_cols_high_card], FUN=as.numeric)
  }
  
  # Numeric variable preprocessing

  # Record numeric columns that have variation
  mnum_cols <- colnames(data)[!(colnames(data) %in% mchar_cols)]

  # Check that we have at least one numeric variable
  if (length(mnum_cols) > 0) {

    # Keep only those columns that are not only a single value 
    mnum_cols_a <- mnum_cols[which(sapply(data[mnum_cols], FUN=function(X) { 
                                        return((is.finite(max(X, na.rm=TRUE))) &
                                                ((max(X, na.rm=TRUE) != min(X, na.rm=TRUE)) | (any(is.na(X)))))}))] 
    if (length(mnum_cols_a) > 0) {
      mnum_cols <- mnum_cols_a
    } else {
      if (length(char_cols_used) ==0 ) {
        mnum_cols <- mnum_cols[1]
      }
    }

  }

  # Create DataFrame 
  data <- data.matrix(data[c(char_cols_used, mnum_cols)])
  
  
  # Ensure that it is possible to fit a tree with min_obs
  if (nrow(data) < 10000) {
    n_train <- round(0.8 * nrow(data))
  } else {
    n_train <- round(0.65 * nrow(data))
  }

  b_frac <- 0.5
  min_obs <- min(10, ceiling((b_frac * n_train - 1) / 2. ** 5 ))

  # Record the preprocessing time
  print("Preprocessing time:")
  print(proc.time() - start_time)

  # Model fitting

  # Fit the model
  mgbm <- gbm.fit(x=data, y=response, distribution=DIST, n.trees=NTREES, shrinkage=SHRINK,
                  interaction.depth=3, bag.fraction=b_frac, nTrain=n_train,
                  keep.data=FALSE, verbose=FALSE, n.minobsinnode=min_obs)
  
  # Find the optimal number of trees using the test fraction
  mopt_ntrees <- gbm.perf(mgbm, method="test", plot.it=FALSE)
  
  # Fallback to OOB if method="test" fails
  if (is.null(mopt_ntrees) || length(mopt_ntrees) == 0) {
    print("Fallback to OOB")
    mopt_ntrees <- gbm.perf(mgbm, method="OOB", plot.it=FALSE)
  }
  
  # Store the model and the information needed for preprocessing at predict time
  model <- list(gbm=mgbm, char_modes=mmodes, factor_maps=mfactor_maps,
                char_cols=mchar_cols, num_cols=mnum_cols,
                char_cols_low_card=mchar_cols_low_card, opt_ntrees=mopt_ntrees,
                char_cols_high_card=mchar_cols_high_card)
  
  return(model)
};

modelpredict <- function(model,data) {
  # Function to make class probablity predictions using a fitted gbm model
  #
  # Args:
  #   model : list
  #     Contains stored state information
  #   data : data.frame
  #     Contains data to make predictions on
  #
  # Returns:
  # list
  #   Contains predicted values for the positive class probablilites

  # Load the packages
  library(gdata)
  library(gbm)

  # Categorical Variable Preprocessing

  # Encode Factor Levels to match the same order as at fit time
  used_char_cols <- union(model$char_cols_low_card, model$char_cols_high_card)
  data[used_char_cols] <- lapply(data[used_char_cols], FUN=as.character)
  
  if (length(used_char_cols) > 1) {
    for(i in 1:length(used_char_cols)) {
      cur_col <-used_char_cols[i]
      
      # Map the factor levels so they are consistent
      mapLevels(data[[cur_col]]) <- model$factor_maps[[cur_col]]
      
      # If new, encode to low support if there, otherwise ignore
      if(cur_col %in% model$char_cols_high_card) {
        if("--Low Support--" %in% levels(data[[cur_col]])) {
          data[[cur_col]][is.na(as.integer(data[[cur_col]]))] <-  which(levels(data[[cur_col]]) == "--Low Support--")
        }
        else {
          data[[cur_col]][is.na(as.integer(data[[cur_col]]))] <-  length(model$factor_maps[[cur_col]]) + 1
        }
      }
    }
    
    # Recode high-cardinality values as numeric
    data[model$char_cols_high_card] <-  lapply(data[model$char_cols_high_card], FUN=as.numeric)
  }
  
  # Model Prediction

  # Create the data Matrix
  data <- data[c(model$char_cols_low_card, model$char_cols_high_card, model$num_cols)]
  
  # Return predictions
  predictions <- predict.gbm(model[['gbm']], newdata=data, type='response', n.trees=model$opt_ntrees)

};

datarobot.run("R Gradient-Boosted Trees Classifier Reference Model", modelfit, modelpredict)
