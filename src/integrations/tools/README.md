### Category
```
Basic Tools
- RAG
- file readers
- file analysis from user command (extract information, get the min max mean values...)

Production/Pressure cruve analysis Tools
- DCA 
- Cumulative production
- Well test, build up, drawdown
- Characerization from the curves
- Pressure transient analysis

Lab sample analysis Tools
- Relative perm
- Composition (WINPROP)
- PVT

Geological analysis Tools
- Interpolation
- Contour
- Export geological mesh

Forecasting Tools
    -Data-driven Model(pre-trained/online training)
        - Production forecasting (Montney)
        - Geo data forecasting (Montney)

    -Plysics-driven
        - Numerical simulation
        - Analytical
        - PINN

### Dependency Relationship
```
las_reader
    |
    --> geostats_interpolation_tools_db
    |       |
    |       --> plot_contour_tool
    |
    --> (if location is within a play) --> geo_pred_pretrained
    |                               |
    |                               --> DL_pred_pretrain
    |
    |
    --> (user uploads production data of wells with LAS and location info)
            |
            --> online_fit (to predict production profile)
                                        
geo_pred_pretrained or DL_pred_pretrain
    |
    --> DCA_tool
    |      |
    |      --> plot_fitted_curve_tool
    |
    --> (if production data is available) --> NPV_tool


