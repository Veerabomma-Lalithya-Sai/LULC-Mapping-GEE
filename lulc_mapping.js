// =====================================================
// STEP 1: Visualize Study Area
// =====================================================
Map.centerObject(StudyArea, 10);
Map.addLayer(
  StudyArea,
  {color: 'red', fillColor: '00000000', width: 2},
  'Study Area (Outline)'
);


// =====================================================
// STEP 2: Cloud Masking Function (Sentinel-2 QA60)
// =====================================================
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
              .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask);
}


// =====================================================
// STEP 3: Load Sentinel-2 Surface Reflectance (HARMONIZED)
// =====================================================
var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(StudyArea)
  .filterDate('2023-10-01', '2023-11-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds)
  .select(['B2', 'B3', 'B4', 'B8']); // Blue, Green, Red, NIR

// Create median composite
var image = sentinel2.median().clip(StudyArea);

// RGB visualization
Map.addLayer(image, {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 3000
}, 'Cloud-free Sentinel-2 RGB');


// =====================================================
// STEP 4: Assign Class Labels (Training Polygons)
// =====================================================
// Geometry Imports required:
// Water, Vegetation, Builtup, Barren (FeatureCollection)

var water = Water.map(function(f) {
  return f.set('class', 0);
});

var vegetation = Vegetation.map(function(f) {
  return f.set('class', 1);
});

var builtup = Builtup.map(function(f) {
  return f.set('class', 2);
});

var barren = Barren.map(function(f) {
  return f.set('class', 3);
});

// Merge all training polygons
var trainingSamples = water
  .merge(vegetation)
  .merge(builtup)
  .merge(barren);

// Show training polygons as outlines only
Map.addLayer(
  trainingSamples,
  {color: 'white', fillColor: '00000000', width: 1},
  'Training Polygons (Outline)'
);

print('Sample training feature:', trainingSamples.first());


// =====================================================
// STEP 5: RANDOM FOREST CLASSIFICATION
// =====================================================

// 5.1 Define input bands
var bands = ['B2', 'B3', 'B4', 'B8'];


// -----------------------------------------------------
// 5.2 Sample pixels from training polygons
// -----------------------------------------------------
var samples = image.select(bands).sampleRegions({
  collection: trainingSamples,
  properties: ['class'],
  scale: 10,
  geometries: false
});

// Add random column for balancing
samples = samples.randomColumn('rand');


// -----------------------------------------------------
// 5.3 Balance samples per class
// -----------------------------------------------------
var waterSamples  = samples.filter(ee.Filter.eq('class', 0)).limit(400, 'rand');
var vegSamples    = samples.filter(ee.Filter.eq('class', 1)).limit(400, 'rand');
var builtSamples  = samples.filter(ee.Filter.eq('class', 2)).limit(400, 'rand');
var barrenSamples = samples.filter(ee.Filter.eq('class', 3)).limit(400, 'rand');

// Final balanced training dataset
var training = waterSamples
  .merge(vegSamples)
  .merge(builtSamples)
  .merge(barrenSamples);

print('Balanced training dataset:', training);


// -----------------------------------------------------
// 5.4 Train Random Forest classifier
// -----------------------------------------------------
var rfClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100
}).train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});


// -----------------------------------------------------
// 5.5 Classify the image
// -----------------------------------------------------
var classified = image.select(bands).classify(rfClassifier);


// -----------------------------------------------------
// 5.6 Display final LULC map
// -----------------------------------------------------
var lulcVis = {
  min: 0,
  max: 3,
  palette: [
    '0000FF', // 0 - Water
    '00FF00', // 1 - Vegetation
    'FF0000', // 2 - Built-up
    'FFFF00'  // 3 - Barren
  ]
};

Map.addLayer(
  classified.clip(StudyArea),
  lulcVis,
  'LULC - Random Forest',
  true,
  0.7   // opacity
);

/// =====================================================
// STEP 6: ACCURACY ASSESSMENT (CORRECT & SAFE)
// =====================================================

// Add random column once
var samplesWithRandom = samples.randomColumn('split');

// Split first
var trainAll = samplesWithRandom.filter(ee.Filter.lt('split', 0.7));
var testAll  = samplesWithRandom.filter(ee.Filter.gte('split', 0.7));

// ---- BALANCE TRAINING SET ----
var trainWater  = trainAll.filter(ee.Filter.eq('class', 0)).limit(200);
var trainVeg    = trainAll.filter(ee.Filter.eq('class', 1)).limit(200);
var trainBuilt  = trainAll.filter(ee.Filter.eq('class', 2)).limit(200);
var trainBarren = trainAll.filter(ee.Filter.eq('class', 3)).limit(200);

var trainSet = trainWater
  .merge(trainVeg)
  .merge(trainBuilt)
  .merge(trainBarren);

// ---- BALANCE TEST SET ----
var testWater  = testAll.filter(ee.Filter.eq('class', 0)).limit(100);
var testVeg    = testAll.filter(ee.Filter.eq('class', 1)).limit(100);
var testBuilt  = testAll.filter(ee.Filter.eq('class', 2)).limit(100);
var testBarren = testAll.filter(ee.Filter.eq('class', 3)).limit(100);

var testSet = testWater
  .merge(testVeg)
  .merge(testBuilt)
  .merge(testBarren);

// Train RF
var rfAccuracy = ee.Classifier.smileRandomForest({
  numberOfTrees: 100
}).train({
  features: trainSet,
  classProperty: 'class',
  inputProperties: bands
});

// Validate
var validated = testSet.classify(rfAccuracy);

// Confusion Matrix
var confusionMatrix = validated.errorMatrix('class', 'classification');
print('Confusion Matrix:', confusionMatrix);

// Accuracy metrics
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa:', confusionMatrix.kappa());

// =====================================================
// STEP 7: AREA CALCULATION (EXPORT-SAFE)
// =====================================================

var areaImage = ee.Image.pixelArea()
  .addBands(classified.rename('class'));

var areaStats = areaImage.reduceRegions({
  collection: ee.FeatureCollection([
    ee.Feature(StudyArea, {})
  ]),
  reducer: ee.Reducer.sum().group({
    groupField: 1,
    groupName: 'class'
  }),
  scale: 50   // coarser scale to avoid timeout
});

Export.table.toDrive({
  collection: areaStats,
  description: 'LULC_Area_Statistics',
  fileFormat: 'CSV'
});


// =====================================================
// END OF SCRIPT
// =====================================================
