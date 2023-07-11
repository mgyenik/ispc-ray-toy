#include <algorithm>
#include <vector>
#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <cmath>

#include <stdint.h>

#include "render_ispc.h"

using namespace ::ispc;
using vec3 = ispc::float3;
// using AABB = ispc::AABB;

float length(vec3 a) {
  return sqrt(a.v[0]*a.v[0] + a.v[1]*a.v[1] + a.v[2]*a.v[2]);
}

inline vec3 mul(vec3 a, vec3 b) {
  return {a.v[0]*b.v[0], a.v[1]*b.v[1], a.v[2]*b.v[2]};
}

inline vec3 mul(float s, vec3 b) {
  return {s*b.v[0], s*b.v[1], s*b.v[2]};
}

inline vec3 add(vec3 a, vec3 b) {
  return {a.v[0]+b.v[0], a.v[1]+b.v[1], a.v[2]+b.v[2]};
}

inline vec3 sub(vec3 a, vec3 b) {
  return {a.v[0]-b.v[0], a.v[1]-b.v[1], a.v[2]-b.v[2]};
}

inline float random_float() {
  // Returns a random real in [0,1).
  return rand() / (RAND_MAX + 1.0);
}

inline float random_float(float min, float max) {
  // Returns a random real in [min,max).
  return min + (max-min)*random_float();
}

inline int random_int(int min, int max) {
  // Returns a random integer in [min,max].
  return static_cast<int>(random_float(min, max+1));
}

inline vec3 random_vec(float dmin, float dmax) {
  return {random_float(dmin, dmax),
          random_float(dmin, dmax),
          random_float(dmin, dmax)};
}

vec3 Center(const MovingSphere& ms, float time) {
  const float dv = (time - ms.time0)/(ms.time1 - ms.time0);
  return add(ms.center0, mul(dv, sub(ms.center1, ms.center0)));
}

AABB SurroundingBox(const AABB& a, const AABB& b) {
  vec3 small = {std::min(a.min.v[0], b.min.v[0]),
                std::min(a.min.v[1], b.min.v[1]),
                std::min(a.min.v[2], b.min.v[2])};
  vec3 big = {std::max(a.max.v[0], b.max.v[0]),
              std::max(a.max.v[1], b.max.v[1]),
              std::max(a.max.v[2], b.max.v[2])};
  return {small, big};
}

AABB BoundingBox(const Sphere& sphere) {
  const float r = sphere.radius;
  const float3 rv= {r,r,r};
  return {sub(sphere.center, rv),
          add(sphere.center, rv)};
}

AABB BoundingBox(const MovingSphere& sphere) {
  const vec3 c0 = Center(sphere, sphere.time0);
  const vec3 c1 = Center(sphere, sphere.time1);
  const float r = sphere.radius;
  const vec3 rv= {r,r,r};
  const AABB box0 = {sub(c0, rv),
                     add(c0, rv)};
  const AABB box1 = {sub(c1, rv),
                     add(c1, rv)};
  return SurroundingBox(box0, box1);
}

AABB BoundingBox(const XYRect& rect) {
  return {{rect.x0, rect.y0, rect.k-0.0001f}, {rect.x1, rect.y1, rect.k-0.0001f}};
}

AABB BoundingBox(const XZRect& rect) {
  return {{rect.x0, rect.k-0.0001f, rect.z0}, {rect.x1, rect.k-0.0001f, rect.z1}};
}

AABB BoundingBox(const YZRect& rect) {
  return {{rect.k-0.0001f, rect.y0, rect.z0}, {rect.k-0.0001f, rect.y1, rect.z1}};
}

AABB BoundingBox(const Box& box) {
  return {box.min, box.max};
}

// AABB BoundingBox(const Translate& box) {
  
// }

// AABB BoundingBox(const Rotate& box) {
  
// }

AABB BoundingBox(const Hittable& h) {
  if (h.type == SPHERE) {
    return BoundingBox(h.sphere);
  }
  if (h.type == MOVING_SPHERE) {
    return BoundingBox(h.moving_sphere);
  }
  if (h.type == XY_RECT) {
    return BoundingBox(h.xy_rect);
  }
  if (h.type == XZ_RECT) {
    return BoundingBox(h.xz_rect);
  }
  if (h.type == YZ_RECT) {
    return BoundingBox(h.yz_rect);
  }
  if (h.type == BOX) {
    return BoundingBox(h.box);
  }
  std::cerr << "UNSUPPORTED AABB TYPE: " << h.type << std::endl;
  return {};
}

const void DumpPpm(std::vector<uint8_t> const& r,
                   std::vector<uint8_t> const& g,
                   std::vector<uint8_t> const& b,
                   int img_w, int img_h) {
  std::cout << "P3\n" << img_w << ' ' << img_h << "\n255\n";
  for (int h = img_h-1; h >= 0; --h) {
    for (int w = 0; w < img_w; ++w) {
      const int idx = h*img_w + w;
      int ir = r[idx];
      int ig = g[idx];
      int ib = b[idx];
      std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }
}


struct Scene {
  int w;
  int h;
  int samples;
  vec3 background;

  vec3 look_from;
  vec3 look_at;
  vec3 vup;
  float aperture;
  float vfov;

  std::vector<ispc::Hittable> hittable;
  std::vector<ispc::Material> material;
  std::vector<ispc::Texture> textures;

  void AddMovingSphere(int material_id, float r, vec3 center0, vec3 center1, float time0, float time1) {
    hittable.push_back({});
    auto& h = hittable.back();
    h.type = ispc::MOVING_SPHERE;
    h.material_id = material_id;
    h.moving_sphere = {center0, center1, time0, time1};
  }

  void AddSphere(int material_id, float r, vec3 center) {
    hittable.push_back({});
    auto& h = hittable.back();
    h.type = ispc::SPHERE;
    h.material_id = material_id;
    h.sphere = {center, r};
  }

  void AddXYRect(int material_id, float x0, float x1, float y0, float y1, float k) {
    hittable.push_back({});
    auto& h = hittable.back();
    h.type = ispc::XY_RECT;
    h.material_id = material_id;
    h.xy_rect = {x0, x1, y0, y1, k};
  }

  void AddXZRect(int material_id, float x0, float x1, float z0, float z1, float k) {
    hittable.push_back({});
    auto& h = hittable.back();
    h.type = ispc::XZ_RECT;
    h.material_id = material_id;
    h.xz_rect = {x0, x1, z0, z1, k};
  }

  void AddYZRect(int material_id, float y0, float y1, float z0, float z1, float k) {
    hittable.push_back({});
    auto& h = hittable.back();
    h.type = ispc::YZ_RECT;
    h.material_id = material_id;
    h.yz_rect = {y0, y1, z0, z1, k};
  }

  void AddBox(int material_id, vec3 p0, vec3 p1) {
    hittable.push_back({});
    auto& h = hittable.back();
    h.type = ispc::BOX;
    h.material_id = material_id;
    h.box = {
             {p0.v[0], p1.v[0], p0.v[1], p1.v[1], p1.v[2]},
             {p0.v[0], p1.v[0], p0.v[1], p1.v[1], p0.v[2]},
             {p0.v[0], p1.v[0], p0.v[2], p1.v[2], p1.v[1]},
             {p0.v[0], p1.v[0], p0.v[2], p1.v[2], p0.v[1]},
             {p0.v[1], p1.v[1], p0.v[2], p1.v[2], p1.v[0]},
             {p0.v[1], p1.v[1], p0.v[2], p1.v[2], p0.v[0]},
             p0, p1,
    };
  }

  int AddLambertian(int texture_id) {
    const int id = material.size();
    material.push_back({ispc::LAMBERTIAN, texture_id, 0, 0.0, 0.0});
    return id;
  }

  int AddLambertian(vec3 albedo) {
    const int tid = AddSolidTexture(albedo);
    const int id = material.size();
    material.push_back({ispc::LAMBERTIAN, tid, 0, 0.0, 0.0});
    return id;
  }

  int AddMetallic(float fuzz, int texture_id) {
    if (fuzz > 1) {
      fuzz = 1;
    }
    const int id = material.size();
    material.push_back({ispc::METALLIC, texture_id, 0, fuzz, 0.0});
    return id;
  }

  int AddMetallic(float fuzz, vec3 albedo) {
    const int tid = AddSolidTexture(albedo);
    if (fuzz > 1) {
      fuzz = 1;
    }
    const int id = material.size();
    material.push_back({ispc::METALLIC, tid, 0, fuzz, 0.0});
    return id;
  }

  int AddDielectric(float eta) {
    const int id = material.size();
    material.push_back({ispc::DIELECTRIC, 0, 0, 0.0, eta});
    return id;
  }

  int AddDiffuseLight(int emit_texture_id) {
    const int id = material.size();
    material.push_back({ispc::DIELECTRIC, 0, emit_texture_id, 0.0, 0.0});
    return id;
  }

  int AddDiffuseLight(vec3 emit_color) {
    const int tid = AddSolidTexture(emit_color);
    const int id = material.size();
    material.push_back({ispc::DIFFUSE_LIGHT, 0, tid, 0.0, 0.0});
    return id;
  }

  int AddSolidTexture(vec3 color) {
    const int id = textures.size();
    textures.push_back({ispc::SOLID, {{color}}, {}});
    return id;
  }

  int AddCheckerTexture(vec3 color1, vec3 color2) {
    const int id = textures.size();
    textures.push_back({ispc::CHECKER, {}, {color1, color2}});
    return id;
  }
};

Scene BuildCornellBoxWorld() {
  Scene world;
  world.look_from = {278, 278, -800};
  world.look_at = {278, 278, 0};
  world.vup = {0, 1, 0};
  world.aperture = 0.1;
  world.vfov = 40.0f;

  world.background = {0,0,0};

  const auto red = world.AddLambertian({0.65, .05, .05});
  const auto white = world.AddLambertian({.73, .73, .73});
  const auto green = world.AddLambertian({.12, .45, .15});
  const auto light = world.AddDiffuseLight({15, 15, 15});

  world.AddYZRect(green, 0, 555, 0, 555, 555);
  world.AddYZRect(red, 0, 555, 0, 555, 0);
  world.AddXZRect(light, 213, 343, 227, 332, 554);
  world.AddXZRect(white, 0, 555, 0, 555, 0);
  world.AddXZRect(white, 0, 555, 0, 555, 555);
  world.AddXYRect(white, 0, 555, 0, 555, 555);

  world.AddBox(white, {130, 0, 65}, {295, 165, 230});
  world.AddBox(white, {265, 0, 295}, {430, 330, 460});

  return world;
}

Scene BuildSimpleLightWorld() {
  Scene world;
  world.look_from = {26, 3, 6};
  world.look_at = {0, 2, 0};
  world.vup = {0, 1, 0};
  world.aperture = 0.1;
  world.vfov = 20.0f;

  // world.background = {0.7f, 0.8f, 1.0f};
  world.background = {0,0,0};

  const int ground = world.AddLambertian({0.5, 0.5, 0.5});
  world.AddSphere(ground, 1000, {0,-1000,0});

  auto material3 = world.AddMetallic(0.0, {0.7, 0.6, 0.5});
  world.AddSphere(material3, 2.0, {0, 2, 0});

  auto light_material = world.AddDiffuseLight({4, 4, 4});
  world.AddXYRect(light_material, 3,5,1,3,-2);
  return world;
}

Scene BuildWeek1World() {
  Scene world;
  world.look_from = {13, 2, 3};
  world.look_at = {0, 0, 0};
  world.vup = {0, 1, 0};
  world.aperture = 0.1;
  world.vfov = 20.0f;

  world.background = {0.7f, 0.8f, 1.0f};
  const int checker_ground = world.AddCheckerTexture({0.2, 0.3, 0.1}, {0.9, 0.9, 0.9});
  const int ground = world.AddLambertian(checker_ground);
  // const int ground = world.AddLambertian({0.5, 0.5, 0.5});
  world.AddSphere(ground, 1000, {0,-1000,0});

  const int glass = world.AddDielectric(1.5);
  // const int right = world.AddMetallic(1.0, {0.8, 0.6, 0.2});

  // world.AddSphere(center, 0.5, {0,0,-1});
  // world.AddSphere(left, 0.5, {-1,0,-1});
  // world.AddSphere(right, 0.5, {1,0,-1});
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = random_float();
      vec3 center = {a + 0.9f*random_float(), 0.2f, b + 0.9f*random_float()};

      // if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
      if (length({center.v[0]-4.0f, center.v[1]-0.2f, center.v[2]-0.0f}) > 0.9f) {
        if (choose_mat < 0.8) {
          // diffuse
          auto albedo = mul(random_vec(0, 1), random_vec(0, 1));
          auto tid = world.AddSolidTexture(albedo);
          auto mid = world.AddLambertian(tid);
          world.AddSphere(mid, 0.2, center);
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = random_vec(0.5, 1);
          auto fuzz = random_float(0, 0.5);
          auto tid = world.AddSolidTexture(albedo);
          auto material_id = world.AddMetallic(fuzz, tid);
          world.AddSphere(material_id, 0.2, center);
        } else {
          // glass
          world.AddSphere(glass, 0.2, center);
        }
      }
    }
  }

  auto material2 = world.AddLambertian({0.4, 0.2, 0.1});
  auto material3 = world.AddMetallic(0.0, {0.7, 0.6, 0.5});
  world.AddSphere(glass, 1.0, {0, 1, 0});
  world.AddSphere(material2, 1.0, {-4, 1, 0});
  world.AddSphere(material3, 1.0, {4, 1, 0});

  return world;
}


ispc::RenderParams ParamsFromScene(const Scene& scene) {
  ispc::RenderParams params;
  params.img_w = scene.w;
  params.img_h = scene.h;
  params.msaa_samples = scene.samples;
  params.time0 = 0;
  params.time1 = 1;
  params.hittable = scene.hittable.data();
  params.size = scene.hittable.size();
  params.material = scene.material.data();
  params.textures = scene.textures.data();
  params.background = scene.background;

  params.look_from = scene.look_from;
  params.look_at = scene.look_at;
  params.vup = scene.vup;
  params.aperture = scene.aperture;
  params.vfov = scene.vfov;

  return params;
}

int AddObjs(const std::vector<Hittable>& objs, int left, int right, std::vector<BVH2>& bvh) {
  const int id = bvh.size();
  bvh.push_back({});
  auto& node = bvh.back();

  node.prim_mask = 0x03;
  node.children[0] = left;
  node.children[1] = right;
  node.bounds = SurroundingBox(BoundingBox(objs[left]),
                               BoundingBox(objs[right]));
  return id;
}

int AddNodes(int left, int right, std::vector<BVH2>& bvh) {
  const int id = bvh.size();
  bvh.push_back({});
  auto& node = bvh.back();

  node.prim_mask = 0x00;
  node.children[0] = left;
  node.children[1] = right;
  node.bounds = SurroundingBox(bvh[left].bounds,
                               bvh[right].bounds);
  return id;
}

inline bool box_compare(const Hittable& a, const Hittable& b, int axis) {
  AABB box_a = BoundingBox(a);
  AABB box_b = BoundingBox(b);
  return box_a.min.v[axis] < box_b.min.v[axis];
}


bool box_x_compare (const Hittable& a, const Hittable& b) {
  return box_compare(a, b, 0);
}

bool box_y_compare (const Hittable& a, const Hittable& b) {
  return box_compare(a, b, 1);
}

bool box_z_compare (const Hittable& a, const Hittable& b) {
  return box_compare(a, b, 2);
}

int BVH2FromObjects(const std::vector<Hittable>& objs, int start, int end, std::vector<BVH2>& bvh) {
  const int span = end - start;
  if (span == 1) {
    return AddObjs(objs, start, start, bvh);
  }

  int axis = random_int(0,2);
  auto comparator = (axis == 0) ? box_x_compare
                  : (axis == 1) ? box_y_compare
                                : box_z_compare;

  if (span == 2) {
    const int other = start+1;
    if (comparator(objs[start], objs[other])) {
      return AddObjs(objs, start, other, bvh);
    } else {
      return AddObjs(objs, other, start, bvh);
    }
  }

  auto objects = objs;
  std::sort(objects.begin() + start, objects.begin() + end, comparator);

  const int mid = start + span/2;
  const int left = BVH2FromObjects(objects, start, mid, bvh);
  const int right = BVH2FromObjects(objects, mid, end, bvh);
  return AddNodes(left, right, bvh);
}

struct BuiltBVH2 {
  int root;
  std::vector<BVH2> nodes;
};

BuiltBVH2 BVH2FromScene(const Scene& scene) {
  BuiltBVH2 bvh;
  bvh.root = BVH2FromObjects(scene.hittable, 0, scene.hittable.size(), bvh.nodes);
  std::cerr << "BVH ROOT: " << bvh.root << std::endl;
  return bvh;
}

int main(int argc, char** argv) {
#define CSR_FLUSH_TO_ZERO         (1 << 15)
  unsigned csr = __builtin_ia32_stmxcsr();
  csr |= CSR_FLUSH_TO_ZERO;
  __builtin_ia32_ldmxcsr(csr);

  auto world = BuildWeek1World();
  // auto world = BuildCornellBoxWorld();
  // const int h = 512;
  // const int w = 1024;
  // const float ratio = 3.0f/2.0f;
  const float ratio = 16.0f/9.0f;
  const int w = 400;
  const int h = static_cast<int>(w/ratio);
  // const int h = 600;
  world.w = w;
  world.h = h;
  world.samples = 200;
  auto params = ParamsFromScene(world);

  std::vector<uint8_t> r(w*h);
  std::vector<uint8_t> g(w*h);
  std::vector<uint8_t> b(w*h);
  params.rbuf = r.data();
  params.gbuf = g.data();
  params.bbuf = b.data();

  BuiltBVH2 bvh;
  {
    auto start = std::chrono::steady_clock::now();
    bvh = BVH2FromScene(world);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Elapsed BVH build time " << elapsed.count() << " milliseconds." << std::endl;
  }

  params.bvh = bvh.nodes.data();
  params.bvh_root = bvh.root;

  {
    auto start = std::chrono::steady_clock::now();
    ispc::render_parallel(params);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Elapsed render time " << elapsed.count() << " milliseconds." << std::endl;
  }
  {
    auto start = std::chrono::steady_clock::now();
    DumpPpm(r, g, b, w, h);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Elapsed PPM time " << elapsed.count() << " milliseconds." << std::endl;
  }
}
