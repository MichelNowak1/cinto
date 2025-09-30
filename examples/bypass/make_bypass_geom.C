// Initialize the geometry manager
TGeoManager *geom = new TGeoManager("bypass", "bypass geometry");

// diffusive media
TGeoMaterial *diffusive_material = new TGeoMaterial("TOP_material", 0, 0, 0);
TGeoMedium *diffusive = new TGeoMedium("TOP", 1, diffusive_material);

// concrete media
TGeoMaterial *concrete_material= new TGeoMaterial("boleft_material", 0, 0, 0);
TGeoMedium *concrete = new TGeoMedium("TOP", 1, concrete_material);

// top volume
TGeoVolume *top_volume= geom->MakeBox("TOP", diffusive, 40, 40, 100);
top_volume->SetLineColor(kBlack);
geom->SetTopVolume(top_volume);

// concrete volume
TGeoVolume *concrete_volume= geom->MakeBox("boleft", concrete, 30, 30, 100);
concrete_volume->SetLineColor(kBlue);
top_volume->AddNode(concrete_volume, 1);

// detector volume
TGeoVolume *detector_volume = geom->MakeBox("detector", diffusive, 5, 5, 100);
detector_volume->SetLineColor(kGreen);
top_volume->AddNode(detector_volume, 2, new TGeoTranslation(35.0, 35.0, 0));

geom->CloseGeometry();
geom->Export("bypass_geom.root");
geom->SetTopVisible();
geom->SetVisLevel(3);

top_volume->Draw("ogl");
