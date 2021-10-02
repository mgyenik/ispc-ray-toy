#include <vector>
#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <cmath>

#include <stdint.h>

#include "render_ispc.h"

using vec3 = ispc::float3;

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
  std::vector<ispc::Hittable> hittable;
  std::vector<ispc::Material> material;

  void AddMovingSphere(int material_id, float r, vec3 center0, vec3 center1, float time0, float time1) {
    hittable.push_back({ispc::MOVING_SPHERE, material_id, {}, {center0, center1, time0, time1, r}});
  }

  void AddSphere(int material_id, float r, vec3 center) {
    hittable.push_back({ispc::SPHERE, material_id, {center, r}, {}});
  }

  int AddLambertian(vec3 albedo) {
    const int id = material.size();
    material.push_back({ispc::LAMBERTIAN, albedo, 0.0, 0.0});
    return id;
  }

  int AddMetallic(float fuzz, vec3 albedo) {
    if (fuzz > 1) {
      fuzz = 1;
    }
    const int id = material.size();
    material.push_back({ispc::METALLIC, albedo, fuzz, 0.0});
    return id;
  }

  int AddDielectric(float eta) {
    const int id = material.size();
    material.push_back({ispc::DIELECTRIC, {0,0,0}, 0.0, eta});
    return id;
  }
};

inline float random_float() {
  // Returns a random real in [0,1).
  return rand() / (RAND_MAX + 1.0);
}

inline float random_float(float min, float max) {
  // Returns a random real in [min,max).
  return min + (max-min)*random_float();
}

inline vec3 random_vec(float dmin, float dmax) {
  return {
          random_float(dmin, dmax),
          random_float(dmin, dmax),
          random_float(dmin, dmax),
  };
}

float length(vec3 a) {
  return sqrt(a.v[0]*a.v[0] + a.v[1]*a.v[1] + a.v[2]*a.v[2]);
}

inline vec3 mul(vec3 a, vec3 b) {
  return {a.v[0]*b.v[0], a.v[1]*b.v[1], a.v[2]*b.v[2]};
}

inline vec3 add(vec3 a, vec3 b) {
  return {a.v[0]+b.v[0], a.v[1]+b.v[1], a.v[2]+b.v[2]};
}

Scene BuildWorld() {
  Scene world;
  const int ground = world.AddLambertian({0.5, 0.5, 0.5});
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
          auto mid = world.AddLambertian(albedo);
          auto center2 = add(center, vec3{0, random_float(0, 0.5), 0});
          world.AddMovingSphere(mid, 0.2, center, center2, 0, 1);
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = random_vec(0.5, 1);
          auto fuzz = random_float(0, 0.5);
          auto material_id = world.AddMetallic(fuzz, albedo);
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


int main(int argc, char** argv) {
  // const int h = 512;
  // const int w = 1024;
  const int msaa_samples = 100;
  const float ratio = 3.0f/2.0f;
  //const float ratio = 16.0f/9.0f;
  const int w = 400;
  const int h = static_cast<int>(w/ratio);
  std::vector<uint8_t> r(w*h);
  std::vector<uint8_t> g(w*h);
  std::vector<uint8_t> b(w*h);

  auto world = BuildWorld();

  {
    auto start = std::chrono::steady_clock::now();
    ispc::render_parallel(world.hittable.data(), world.hittable.size(),
                 world.material.data(),
                 w, h,
                0, 1,
                msaa_samples, r.data(), g.data(), b.data());
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
