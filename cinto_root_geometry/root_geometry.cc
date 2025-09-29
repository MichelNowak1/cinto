#include <iostream>
#include <thread>
#include <chrono>
#include "TGeoManager.h"
#include "TThread.h"
#include "TGeoNavigator.h"
#include "TNode.h"
#include "TError.h"
#include "TGeoBBox.h"

static std::vector<int> node_ids;

extern "C" int get_node_ids_size(){
    return node_ids.size();
}

extern "C" int get_node_id(){
    int node_id = node_ids.back();
    node_ids.pop_back();
    return node_id;
}

extern "C" void init_thread()
{
    std::thread::id this_id = std::this_thread::get_id();
    auto *nav = gGeoManager->GetCurrentNavigator();
    if (!nav) {
        gGeoManager->AddNavigator();
    }
}
extern "C" void init_root_geometry(const char* file_name)
{
    gErrorIgnoreLevel = 1001;
    gGeoManager->Import(file_name);
    gGeoManager->SetMaxThreads(12);
}

extern "C" double* get_geometry_coordinates()
{
    auto top_volume = (TGeoBBox*) gGeoManager->GetTopVolume()->GetShape();
    double *origin = (double*) top_volume->GetOrigin();
    double dx = (double) top_volume->GetDX();
    double dy = (double) top_volume->GetDY();
    double dz = (double) top_volume->GetDZ();
    double *coordinates = new double[6];
    coordinates[0] = origin[0] - dx,
    coordinates[1] = origin[0] + dx,
    coordinates[2] = origin[1] - dy,
    coordinates[3] = origin[1] + dy,
    coordinates[4] = origin[2] - dz,
    coordinates[5] = origin[2] + dz;
    return coordinates;
}

extern "C" const double get_volume_capacity(double *position)
{
    auto *nav = gGeoManager->GetCurrentNavigator();
    
    nav->SetCurrentPoint(position);
    TGeoNode *node = nav->FindNode();
    
    double volume_capacity = nav->GetCurrentNode()->GetVolume()->Capacity();
    return volume_capacity;
}


extern "C" const char* get_volume_name(double *position)
{
    auto *nav = gGeoManager->GetCurrentNavigator();

    nav->SetCurrentPoint(position);
    TGeoNode *node = nav->FindNode();

    const char* volume_name = nav->GetCurrentNode()->GetVolume()->GetName();

    return volume_name;
}

extern "C" void set_position_and_direction(double *position, double* direction)
{
    auto *nav = gGeoManager->GetCurrentNavigator();
    nav->SetCurrentPoint(position);
    nav->InitTrack(position, direction);
}

extern "C" void set_direction(double* direction)
{
    auto *nav = gGeoManager->GetCurrentNavigator();
    nav->SetCurrentDirection(direction);
}
extern "C" unsigned int get_material(double *position, int thread_number)
{
    auto *nav = gGeoManager->GetCurrentNavigator();
    nav->SetCurrentPoint(position);
    TGeoNode *node = nav->FindNode();

    unsigned int material_index = nav->GetCurrentNodeId();

    return material_index;
}

extern "C" double get_distance_to_next_boundary(double *position, double *direction)
{
    auto *nav = gGeoManager->GetCurrentNavigator();
    nav->InitTrack(position, direction);
    nav->FindNextBoundary();
    double distance_to_next_frontier = nav->GetStep();
    nav->Step();
    return distance_to_next_frontier;
}

extern "C" bool is_outside_of_geometry(double* position) {
    auto *nav = gGeoManager->GetCurrentNavigator();
    nav->SetCurrentPoint(position);
    return nav->IsOutside();
}

extern "C" double* get_normal_on_boundary(const double *position, const double *direction)
{
    auto *nav = gGeoManager->GetCurrentNavigator();
    nav->InitTrack(position, direction);
    nav->FindNextBoundary();
    double *normal = new double[3];
    normal = nav->FindNormalFast();
    return normal;
}

extern "C" double* step_and_locate(double jump_length) {
    auto *nav = gGeoManager->GetCurrentNavigator();
    nav->Step(jump_length);

    double *new_position = new double[3];
    auto point = nav->GetCurrentPoint();
    new_position[0] = point[0];
    new_position[1] = point[1];
    new_position[2] = point[2];
    return new_position;
}
extern "C" double* cross_boundary_and_locate(double jump_length) {
    auto *nav = gGeoManager->GetCurrentNavigator();
    nav->FindNextBoundary(jump_length);
    nav->CrossBoundaryAndLocate(false, 0);

    double *new_position = new double[3];
    auto point = nav->GetCurrentPoint();
    new_position[0] = point[0];
    new_position[1] = point[1];
    new_position[2] = point[2];
    return new_position;
}

int main(){
}
