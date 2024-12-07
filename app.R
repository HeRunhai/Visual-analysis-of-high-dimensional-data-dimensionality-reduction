library(shiny)
library(ggplot2)
library(plotly)
library(Rtsne)
library(shinydashboard)
library(DT)
library(scales)
library(dplyr)
library(tidyr)
library(trelliscopejs)

# Load and preprocess the data
load_breast_cancer_data <- function() {
  data <- read.csv('Diagnostic Wisconsin Breast Cancer.csv')
  
  # Remove 'id' column and separate diagnosis
  data_features <- data |>
    select(-c(id, diagnosis)) |>
    mutate_all(~ ifelse(is.na(.), mean(., na.rm = TRUE), .)) |> # 填充缺失值
    select_if(~ var(.) > 0) |> # 移除零方差列
    scale()  # Standardize features
  
  # Create diagnosis factor
  diagnosis <- factor(
    data$diagnosis,
    levels = c('M', 'B'),
    labels = c('Malignant', 'Benign')
  )
  
  return(list(features = as.matrix(data_features), diagnosis = diagnosis))
}

load_diabetes_data <- function() {
  data <- read.csv('diabetes.csv')
  
  data_features <- data |>
    select(-c(Outcome)) |>
    scale()
  
  outcome <- factor(
    data$Outcome,
    levels = c('0', '1'),
    labels = c('Not infected', 'Infected')
  )
  
  return(list(features = as.matrix(data_features), outcome = outcome))
}

load_parkinsons_data <- function(){
  data <- read.csv('parkinsons.data')
  
  # Remove the 'name' column and separate the 'status' column
  data_features <- data |>
    select(-c(name, status)) |>
    mutate_all(~ ifelse(is.na(.), mean(., na.rm = TRUE), .)) |>  # Handle missing values
    scale() 
  
  # Create the status factor (1 for Parkinson's disease, 0 for normal)
  status <- factor(
    data$status,
    levels = c(0, 1),
    labels = c('Normal', 'Parkinson\'s Disease')
  )
  
  return(list(features = as.matrix(data_features), status = status))
}

load_recognition_of_handwritten_digits <- function(){
  data <- read.csv('optdigits.tra', header = FALSE, sep = ',')
  
  # The dataset has 64 features and 1 label column
  data_features <- data[, 1:64]
  labels <- data[, 65]
  
  data_features[] <- lapply(data_features, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))  # Handle missing values
  # Remove columns with zero variance
  data_features <- data_features[, apply(data_features, 2, var) > 0]
  
  labels <- factor(labels, levels = 0:9, labels = as.character(0:9))
  
  return(list(features = as.matrix(data_features), labels = labels))
}

load_image_segmentation <- function(){}

load_bank_marketing <- function(){}

load_german_credit <- function(){}

breast_cancer_data <- load_breast_cancer_data()
diabetes_data <- load_diabetes_data()
parkinsons_data <- load_parkinsons_data()
recognition_of_handwritten_digits <- load_recognition_of_handwritten_digits()
image_segmentation <- load_image_segmentation()
bank_marketing <- load_bank_marketing()
german_credit <- load_german_credit()

# Define UI with shinydashboard for a more modern look
ui <- dashboardPage(
  dashboardHeader(
    title = tags$div(style = "color: black; font-weight: bold; font-size: 16px;", "Dimensionality Reduction")
  ),
  dashboardSidebar(sidebarMenu(
    # Dataset and Visualization Type Selection
    tags$div(
      style = "padding: 10px; background-color: #f4f4f4; border-radius: 5px; margin: 10px;",
      tags$style(".control-label {color: black;} .irs-grid-text {color: black;}"),
      selectInput(
        "dataset",
        "Select Dataset:",
        choices = c(
          "Breast Cancer Data",
          "Diabetes Data",
          "Parkinsons Data",
          "Recognition of Handwritten Digits",
          "Image Segmentation",
          "Bank Marketing",
          "German Credit"
        ),
        width = "100%"
      ),
      selectInput(
        "visType",
        "Select Visualization Type:",
        choices = c(
          "Scatter Plot",
          "Density Plot",
          "3D Scatter",
          "Elbow Plot",
          "Variance Heatmap"
        ),
        width = "100%"
      )
    ),
    
    # Interactive Parameter Adjustment
    tags$div(
      style = "padding: 10px; background-color: #e9ecef; border-radius: 5px; margin: 10px; text-align: center;",
      tags$h4(strong("Parameter Adjustment"), style = "color: #495057; margin-bottom: 10px; word-wrap: break-word;"),
      conditionalPanel(
        condition = "input.visType == 'Scatter Plot' || input.visType == '3D Scatter' || input.visType == 'Density Plot'",
        sliderInput(
          "perplexity",
          "t-SNE Perplexity:",
          min = 5,
          max = 50,
          value = 9,
          step = 1,
          width = "100%"
        )
      ),
      conditionalPanel(
        condition = "input.visType == 'Elbow Plot' || input.visType == 'Variance Heatmap'",
        sliderInput(
          "varianceThreshold",
          "Variance Threshold (%):",
          min = 50,
          max = 100,
          value = 90,
          step = 1
        )
      )
    ),
    
    # Features Analysis Buttons
    tags$div(
      style = "padding: 5px; background-color: #f4f4f4; border-radius: 8px; margin: 10px; text-align: center;",
      tags$h4(strong("Features Analysis"), style = "color: #495057; margin-bottom: 15px;"),
      fluidRow(column(
        10,
        actionButton("btnFeatures", "Feature Distributions", style = "width: 100%; height: 50px; background-color: #3498db; color: white; border: none; border-radius: 8px;")
      ), column(
        10,
        actionButton("btnCorrMatrix", "Correlation Matrix", style = "width: 100%; height: 50px; background-color: #2ecc71; color: white; border: none; border-radius: 8px;")
      ))
    ),
    
    # Add Author Information at the bottom
    tags$div(
      style = "background-color: transparent; text-align: center;
           color: white; width: 100%; font-weight: bold;",
      "© 2024 Cai Airong"
    )
  )),
  dashboardBody(uiOutput("dynamicContent"))
)

# Define Server
server <- function(input, output, session) {
  
  # Reactive to calculate explained variance and determine threshold PCs
  explained_variance <- reactive({
    data <- getData()
    if (!is.matrix(data) && !is.data.frame(data)) return(NULL)
    pcaResult <- prcomp(data, scale. = TRUE)
    variance <- (pcaResult$sdev ^ 2) / sum(pcaResult$sdev ^ 2)
    cumsum(variance) * 100
  })
  
  reactive_pcs <- reactive({
    req(input$varianceThreshold)
    cumsum_var <- explained_variance()
    pcs_to_keep <- which(cumsum_var >= input$varianceThreshold)[1]
    list(pcs_to_keep = pcs_to_keep, cumsum_var = cumsum_var)
  })
  
  
  # Reactive dataset loader
  getData <- reactive({
    switch(
      input$dataset,
      "Breast Cancer Data" = breast_cancer_data$features,
      "Diabetes Data" = diabetes_data$features,
      "Parkinsons Data" = parkinsons_data$features,
      "Recognition of Handwritten Digits" = recognition_of_handwritten_digits$features,
      "Image Segmentation" = image_segmentation$features,
      "Bank Marketing" = bank_marketing$features,
      "German Credit" = german_credit$features
    )
  })
  
  # Error message for t-SNE visualization
  output$tsneVisOutput <- renderUI({
    data <- getData()
    perplexity <- input$perplexity
    
    if (perplexity >= nrow(data) / 3) {
      div(
        style = "background-color: #f8d7da; color: #721c24;
                 padding: 15px; border-radius: 5px;
                 text-align: center; margin: 20px;",
        tags$h4("⚠️ Perplexity Too Large"),
        tags$p("Please reduce perplexity to be less than", nrow(data) /
                 3)
      )
    }  else if (input$visType == "Elbow Plot") {
      div(
        style = "background-color: #fff3cd; color: #856404;
                 padding: 15px; border-radius: 5px;
                 text-align: center; margin: 20px;",
        tags$h4("⚠️ Elbow plot is Not Available for t-SNE"),
        tags$p(
          "This visualization type is not supported for the selected algorithm."
        )
      )
    } else {
      plotlyOutput("tsnePlot", height = "400px")
    }
  })
  
  # Default content for PCA and t-SNE
  output$dynamicContent <- renderUI({
    fluidRow(
      box(
        width = 6,
        status = "primary",
        solidHeader = TRUE,
        plotlyOutput("pcaPlot", height = "400px")
      ),
      box(
        width = 6,
        status = "primary",
        solidHeader = TRUE,
        uiOutput("tsneVisOutput", height = "400px")
      ),
      fluidRow(
        box(
          width = 6,
          status = "info",
          DTOutput("pcaParams")
        ),
        box(
          width = 6,
          status = "info",
          DTOutput("tsneParams")
        )
      ),
    )
  })
  
  # PCA Visualization
  output$pcaPlot <- renderPlotly({
    data <- getData()
    
    if (!is.matrix(data) && !is.data.frame(data)) {
      return(NULL)
    }
    
    pcaResult <- prcomp(data, scale. = TRUE)
    explainedVariance <- (pcaResult$sdev ^ 2) / sum(pcaResult$sdev ^ 2)
    cumulativeVar <- cumsum(explainedVariance) * 100
    
    pcs_to_keep <- reactive_pcs()$pcs_to_keep
    visData <- as.data.frame(pcaResult$x[, 1:pcs_to_keep])
    
    plot <- switch(
      input$visType,
      "Scatter Plot" = ggplot(visData, aes(x = PC1, y = PC2)) +
        geom_point(color = "#3498db", alpha = 0.7) +
        labs(
          x = paste0("PC1 (", percent(explainedVariance[1], accuracy = 0.1), ")"),
          y = paste0("PC2 (", percent(explainedVariance[2], accuracy = 0.1), ")")
        ) +
        theme_minimal(),
      
      "Density Plot" = ggplot(visData, aes(PC1, PC2)) +
        geom_density_2d() +
        theme_minimal(),
      
      "Elbow Plot" = ggplot(
        data.frame(
          Components = 1:length(explainedVariance),
          Variance = explainedVariance
        ),
        aes(x = Components, y = Variance)
      ) +
        labs(y="Percentage of explained variances") +
        geom_line(color = "#e74c3c") +
        geom_point(color = "#3498db") +
        theme_minimal(),
      
      "Variance Heatmap" = ggplot(visData, aes(PC1, PC2)) +
        stat_density_2d(
          geom = "raster",
          aes(fill = after_stat(density)),
          contour = FALSE
        ) +
        scale_fill_viridis_c(option = "plasma") +
        theme_minimal()
    )
    
    if (input$visType == "3D Scatter") {
      numPC <- 3
      visData <- as.data.frame(pcaResult$x[, 1:numPC])
      
      plot <- plot_ly(
        data = visData,
        x = ~ PC1,
        y = ~ PC2,
        z = ~ PC3,
        type = "scatter3d",
        mode = "markers",
        marker = list(
          size = 4,
          color = "#3498db",
          opacity = 0.7
        )
      ) |>
        layout(scene = list(
          xaxis = list(title = paste0(
            "PC1 (", percent(explainedVariance[1], accuracy = 0.1), ")"
          )),
          yaxis = list(title = paste0(
            "PC2 (", percent(explainedVariance[2], accuracy = 0.1), ")"
          )),
          zaxis = list(title = paste0(
            "PC3 (", percent(explainedVariance[3], accuracy = 0.1), ")"
          ))
        ))
    }
    
    ggplotly(plot)
  })
  
  output$pcaParams <- renderDT({
    data <- getData()
    pca <- prcomp(data)
    explainedVariance <- (pca$sdev ^ 2) / sum(pca$sdev ^ 2)
    
    data.frame(
      Component = paste0("PC", 1:length(explainedVariance)),
      `Explained Variance (%)` = percent(explainedVariance, accuracy = 0.01)
    ) |>
      datatable(
        options = list(
          pageLength = 5,
          searching = FALSE,
          lengthChange = FALSE
        ),
        class = "cell-border stripe compact"
      )
  })
  
  # t-SNE Visualization
  output$tsnePlot <- renderPlotly({
    # req(input$visType != "3D Scatter")
    data <- getData()
    perplexity <- max(5, min(input$perplexity, nrow(data) / 3))
    
    if (!is.matrix(data) && !is.data.frame(data)) {
      return(NULL)
    }
    
    tsneResult <- Rtsne(data, perplexity = perplexity, dims = 2)$Y
    visData <- as.data.frame(tsneResult)
    names(visData) <- c("V1", "V2")
    
    plot <- switch(
      input$visType,
      "Scatter Plot" = ggplot(visData, aes(V1, V2)) +
        geom_point(color = "#e74c3c", alpha = 0.7) +
        labs(x = "Dimension 1", y = "Dimension 2") +
        theme_minimal(),
      
      "Density Plot" = ggplot(visData, aes(V1, V2)) +
        geom_density_2d() +
        theme_minimal(),
      
      "Variance Heatmap" = ggplot(visData, aes(V1, V2)) +
        stat_density_2d(
          geom = "raster",
          aes(fill = after_stat(density)),
          contour = FALSE
        ) +
        scale_fill_viridis_c(option = "plasma") +
        theme_minimal()
    )
    
    if (input$visType == "3D Scatter") {
      tsneResult <- Rtsne(data, perplexity = perplexity, dims = 3)$Y
      visData <- as.data.frame(tsneResult)
      names(visData) <- c("V1", "V2", "V3")
      
      plot <- plot_ly(
        data = visData,
        x = ~ V1,
        y = ~ V2,
        z = ~ V3,
        type = "scatter3d",
        mode = "markers",
        marker = list(
          size = 4,
          color = "#e74c3c",
          opacity = 0.7
        )
      ) |>
        layout(scene = list(
          xaxis = list(title = "Dimension 1"),
          yaxis = list(title = "Dimension 2"),
          zaxis = list(title = "Dimension 3")
        ))
    }
    
    ggplotly(plot)
  })
  
  output$tsneParams <- renderDT({
    data <- getData()
    perplexity <- max(5, min(input$perplexity, nrow(data) / 3))
    
    data.frame(
      Parameter = c("Perplexity", "Output Dimensions"),
      Value = c(perplexity, 2)
    ) |>
      datatable(
        options = list(
          pageLength = 5,
          searching = FALSE,
          lengthChange = FALSE
        ),
        class = "cell-border stripe compact"
      )
  })
  
  # Feature Distributions Visualization
  observeEvent(input$btnFeatures, {
    output$dynamicContent <- renderUI({
      req(getData())
      
      # Create a box with the visualization
      box(
        width = 12,
        title = tags$div(style = "text-align: center; font-weight: bold; color: #2c3e50;", "Feature Distributions"),
        status = "primary",
        solidHeader = TRUE,
        plotlyOutput("featuresDistributionPlot", height = "550px")
      )
    })
    
    # Render the plotly output for feature distributions
    output$featuresDistributionPlot <- renderPlotly({
      data <- getData()
      
      # Convert matrix to data frame
      df <- as.data.frame(data)
      
      # Gather data for plotting
      df_long <- df |>
        pivot_longer(everything(),
                     names_to = "Feature",
                     values_to = "Value")
      
      # Create subplot for feature distributions
      plot_list <- lapply(names(df), function(col) {
        plot_ly(
          df,
          x = ~ get(col),
          type = "histogram",
          name = col,
          marker = list(
            color = "#3498db",
            line = list(color = "white", width = 1)
          )
        ) |>
          layout(
            title = col,
            xaxis = list(title = col),
            yaxis = list(title = "Count"),
            bargap = 0.1
          )
      })
      
      # Combine subplots
      subplot(
        plot_list,
        nrows = ceiling(length(plot_list) / 3),
        titleX = TRUE,
        titleY = TRUE
      ) |>
        layout(
          title = "Feature Distributions",
          height = 550,
          margin = list(
            b = 50,
            l = 50,
            r = 50,
            t = 50
          )
        )
    })
  })
  
  
  # Correlation Matrix Visualization
  observeEvent(input$btnCorrMatrix, {
    output$dynamicContent <- renderUI({
      req(getData())
      
      # Create a box with the visualization
      box(
        width = 12,
        title = tags$div(style = "text-align: center; font-weight: bold; color: #2c3e50;", "Correlation Matrix"),
        status = "primary",
        solidHeader = TRUE,
        plotlyOutput("correlationMatrixPlot", height = "550px")
      )
    })
    
    # Render the plotly output for correlation matrix
    output$correlationMatrixPlot <- renderPlotly({
      data <- getData()
      
      # Calculate correlation matrix
      corr_matrix <- cor(data)
      
      # Create heatmap
      plot_ly(
        x = colnames(corr_matrix),
        y = colnames(corr_matrix),
        z = corr_matrix,
        type = "heatmap",
        colors = colorRamp(c("#d73027", "#fee08b", "#1a9850")),
        colorbar = list(title = "Correlation")
      ) |>
        layout(
          title = "Feature Correlation Matrix",
          xaxis = list(title = "", tickangle = 45),
          yaxis = list(title = ""),
          height = 550,
          margin = list(
            b = 100,
            l = 100,
            r = 50,
            t = 50
          )
        )
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)