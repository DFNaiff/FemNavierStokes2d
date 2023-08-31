#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
results_choked_long_100vtu = XMLUnstructuredGridReader(FileName=['/home/dfnaiff/Danilo/Mecanica/TransCal/ToGit/notebooks/results_ns/results_choked_long_10.0.vtu'])
results_choked_long_100vtu.PointArrayStatus = ['ux', 'uy', 'p', 'ux_s', 'uy_s', 'p_s', 'u', 'us', 'dux', 'duy', 'du']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [971, 338]

# show data in view
results_choked_long_100vtuDisplay = Show(results_choked_long_100vtu, renderView1)
# trace defaults for the display properties.
results_choked_long_100vtuDisplay.Representation = 'Surface'
results_choked_long_100vtuDisplay.ColorArrayName = [None, '']
results_choked_long_100vtuDisplay.OSPRayScaleArray = 'du'
results_choked_long_100vtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
results_choked_long_100vtuDisplay.SelectOrientationVectors = 'None'
results_choked_long_100vtuDisplay.ScaleFactor = 1.2000000000000002
results_choked_long_100vtuDisplay.SelectScaleArray = 'None'
results_choked_long_100vtuDisplay.GlyphType = 'Arrow'
results_choked_long_100vtuDisplay.GlyphTableIndexArray = 'None'
results_choked_long_100vtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
results_choked_long_100vtuDisplay.PolarAxes = 'PolarAxesRepresentation'
results_choked_long_100vtuDisplay.ScalarOpacityUnitDistance = 0.6954487164415241

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [6.0, 0.0, 10000.0]
renderView1.CameraFocalPoint = [6.0, 0.0, 0.0]

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator1 = Calculator(Input=results_choked_long_100vtu)
calculator1.Function = ''

# Properties modified on calculator1
calculator1.Function = ''

# show data in view
calculator1Display = Show(calculator1, renderView1)
# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'
calculator1Display.ColorArrayName = [None, '']
calculator1Display.OSPRayScaleArray = 'du'
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.SelectOrientationVectors = 'None'
calculator1Display.ScaleFactor = 1.2000000000000002
calculator1Display.SelectScaleArray = 'None'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'None'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.PolarAxes = 'PolarAxesRepresentation'
calculator1Display.ScalarOpacityUnitDistance = 0.6954487164415241

# hide data in view
Hide(results_choked_long_100vtu, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on calculator1
calculator1.Function = 'ux*iHat + uj*jHat'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on calculator1
calculator1.ResultArrayName = 'uvec'

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator1Display, ('POINTS', 'u', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
calculator1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'u'
uLUT = GetColorTransferFunction('u')

# create a new 'Glyph'
glyph1 = Glyph(Input=calculator1,
    GlyphType='Arrow')
glyph1.Scalars = ['POINTS', 'None']
glyph1.Vectors = ['POINTS', 'None']
glyph1.ScaleFactor = 1.2000000000000002
glyph1.GlyphTransform = 'Transform2'

# Properties modified on glyph1
glyph1.ScaleFactor = 1.2

# show data in view
glyph1Display = Show(glyph1, renderView1)
# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = ['POINTS', 'u']
glyph1Display.LookupTable = uLUT
glyph1Display.OSPRayScaleArray = 'du'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'None'
glyph1Display.ScaleFactor = 1.3199999809265137
glyph1Display.SelectScaleArray = 'None'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'None'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.PolarAxes = 'PolarAxesRepresentation'

# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.GlyphType = '2D Glyph'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleMode = 'vector'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleMode = 'scalar'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleMode = 'vector_components'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleMode = 'off'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleMode = 'vector_components'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleMode = 'vector'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 0.2

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 1.2

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 0.6

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 1.2

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 0.6

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 0.2

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(calculator1)

# Properties modified on calculator1
calculator1.Function = 'ux*iHat + uy*jHat'

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(glyph1)

# Properties modified on glyph1
glyph1.Vectors = ['POINTS', 'uvec']

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 0.2496920683601817

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(calculator1)

# create a new 'Contour'
contour1 = Contour(Input=calculator1)
contour1.ContourBy = ['POINTS', 'dux']
contour1.Isosurfaces = [0.15067011779526573]
contour1.PointMergeMethod = 'Uniform Binning'

# show data in view
contour1Display = Show(contour1, renderView1)
# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName = ['POINTS', 'u']
contour1Display.LookupTable = uLUT
contour1Display.OSPRayScaleArray = 'du'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'uvec'
contour1Display.ScaleFactor = 0.27646487390090674
contour1Display.SelectScaleArray = 'None'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'None'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.PolarAxes = 'PolarAxesRepresentation'

# hide data in view
Hide(calculator1, renderView1)

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(calculator1)

# hide data in view
Hide(contour1, renderView1)

# show data in view
calculator1Display = Show(calculator1, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# destroy contour1
Delete(contour1)
del contour1

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Stream Tracer'
streamTracer1 = StreamTracer(Input=calculator1,
    SeedType='High Resolution Line Source')
streamTracer1.Vectors = ['POINTS', 'uvec']
streamTracer1.MaximumStreamlineLength = 12.0

# init the 'High Resolution Line Source' selected for 'SeedType'
streamTracer1.SeedType.Point1 = [0.0, -1.0, 0.0]
streamTracer1.SeedType.Point2 = [12.0, 1.0, 0.0]

# show data in view
streamTracer1Display = Show(streamTracer1, renderView1)
# trace defaults for the display properties.
streamTracer1Display.Representation = 'Surface'
streamTracer1Display.ColorArrayName = ['POINTS', 'u']
streamTracer1Display.LookupTable = uLUT
streamTracer1Display.OSPRayScaleArray = 'AngularVelocity'
streamTracer1Display.OSPRayScaleFunction = 'PiecewiseFunction'
streamTracer1Display.SelectOrientationVectors = 'Normals'
streamTracer1Display.ScaleFactor = 1.1999767723013066
streamTracer1Display.SelectScaleArray = 'AngularVelocity'
streamTracer1Display.GlyphType = 'Arrow'
streamTracer1Display.GlyphTableIndexArray = 'AngularVelocity'
streamTracer1Display.DataAxesGrid = 'GridAxesRepresentation'
streamTracer1Display.PolarAxes = 'PolarAxesRepresentation'

# hide data in view
Hide(calculator1, renderView1)

# show color bar/color legend
streamTracer1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on streamTracer1.SeedType
streamTracer1.SeedType.Resolution = 100

# Properties modified on streamTracer1.SeedType
streamTracer1.SeedType.Resolution = 100

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(calculator1)

# show data in view
calculator1Display = Show(calculator1, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [6.0, 0.0, 10000.0]
renderView1.CameraFocalPoint = [6.0, 0.0, 0.0]
renderView1.CameraParallelScale = 2.345168274749875

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).