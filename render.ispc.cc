const uniform float pi = 0x1.921fb54442d18p+1; // 3.1415926535...

typedef float<3> float3;

typedef float3 color;

static inline float random_range(varying RNGState *uniform rng, float fmin,
                                 float fmax) {
  return fmin + (fmax - fmin) * frandom(rng);
}

static inline float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float len(float3 a) { return sqrt(dot(a, a)); }

static inline float3 unit(float3 a) { return a / len(a); }

static inline bool near_zero(float3 a) {
  const float s = 1e-8;
  return (abs(a.x) < s) && (abs(a.y) < s) && (abs(a.z) < s);
}

static inline uniform float dot(uniform float3 a, uniform float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline uniform float len(uniform float3 a) { return sqrt(dot(a, a)); }

static inline uniform float3 cross(uniform float3 a, uniform float3 b) {
  uniform float3 r;
  r.x = a.y * b.z - a.z * b.y;
  r.y = a.z * b.x - a.x * b.z;
  r.z = a.x * b.y - a.y * b.x;
  return r;
}

static inline uniform float3 unit(uniform float3 a) { return a / len(a); }

// Generates a seed for a random number generator from 2 inputs plus a backoff
// https://github.com/nvpro-samples/optix_prime_baking/blob/master/random.h
// https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
static inline unsigned int make_seed(unsigned int val0, unsigned int val1) {
  unsigned int v0 = val0, v1 = val1, s0 = 0;
  for (unsigned int n = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }
  return v0;
}

static inline float3 random_vec(varying RNGState *uniform rng, float fmin,
                                float fmax) {
  float3 v = {random_range(rng, fmin, fmax), random_range(rng, fmin, fmax),
              random_range(rng, fmin, fmax)};
  return v;
}

static inline float3 random_in_unit_sphere(varying RNGState *uniform rng) {
  while (true) {
    float3 v = random_vec(rng, -1, 1);
    if (dot(v, v) < 1) {
      return v;
    }
  }
}

static inline float3 random_unit_vec(varying RNGState *uniform rng) {
  return unit(random_in_unit_sphere(rng));
}

static inline float3 random_in_unit_disk(varying RNGState *uniform rng) {
  while (true) {
    float3 v = {random_range(rng, -1, 1), random_range(rng, -1, 1), 0.0};
    if (dot(v, v) < 1) {
      return v;
    }
  }
}

struct Ray {
  float3 orig;
  float3 dir;
  float time;
};

static inline float3 at(const Ray &r, float t) { return r.orig + t * r.dir; }

struct Camera {
  float3 origin;
  float3 lower_left;
  float3 horizontal;
  float3 vertical;
  float3 w;
  float3 u;
  float3 v;
  float lens_radius;
  float time0;
  float time1;
};

uniform Camera MakeCamera(uniform float3 look_from, uniform float3 look_at,
                          uniform float3 vup, uniform float vfov,
                          uniform float aspect_ratio, uniform float aperture,
                          uniform float focus_dist, uniform float time0,
                          uniform float time1) {
  const uniform float theta = pi * vfov / 180.0;
  const uniform float th = tan(theta / 2);
  const uniform float height = 2.0 * th;
  const uniform float width = aspect_ratio * height;

  const uniform float3 w = unit(look_from - look_at);
  const uniform float3 u = unit(cross(vup, w));
  const uniform float3 v = cross(w, u);

  uniform Camera camera;
  camera.w = w;
  camera.u = u;
  camera.v = v;
  camera.origin = look_from;
  camera.horizontal = focus_dist * width * u;
  camera.vertical = focus_dist * height * v;
  camera.lower_left = camera.origin - camera.horizontal / 2 -
                      camera.vertical / 2 - focus_dist * w;
  camera.lens_radius = aperture / 2;
  camera.time0 = time0;
  camera.time1 = time1;
  return camera;
}

Ray get_ray(const uniform Camera &camera, varying RNGState *uniform rng,
            float s, float t) {
  const float3 rd = camera.lens_radius * random_in_unit_disk(rng);
  const float3 offset = camera.u * rd.x + camera.v * rd.y;

  Ray r;
  r.orig = camera.origin + offset;
  r.dir = camera.lower_left + s * camera.horizontal + t * camera.vertical -
          camera.origin - offset;
  r.time = random_range(rng, camera.time0, camera.time1);
  return r;
}

struct HitInfo {
  bool hit;
  float3 p;
  float3 normal;
  float u;
  float v;
  float t;
  bool front_face;
  int material_id;
};

struct Sphere {
  float3 center;
  float radius;
};

struct MovingSphere {
  float3 center0;
  float3 center1;
  float time0;
  float time1;
  float radius;
};

enum HittableType {
  SPHERE,
  MOVING_SPHERE,
  XY_RECT,
  XZ_RECT,
  YZ_RECT,
  BOX,
  TRANSLATE,
  ROTATE_Y,
};

struct XYRect {
  float x0;
  float x1;
  float y0;
  float y1;
  float k;
};

struct XZRect {
  float x0;
  float x1;
  float z0;
  float z1;
  float k;
};

struct YZRect {
  float y0;
  float y1;
  float z0;
  float z1;
  float k;
};

struct Box {
  XYRect s0;
  XYRect s1;
  XZRect s2;
  XZRect s3;
  YZRect s4;
  YZRect s5;
  float3 min;
  float3 max;
};

struct Translate {};

struct RotateY {};

struct Hittable {
  HittableType type;
  int material_id;
  Sphere sphere;
  MovingSphere moving_sphere;
  XYRect xy_rect;
  XZRect xz_rect;
  YZRect yz_rect;
  Box box;
  Translate translate;
  RotateY rotate_y;
};

enum TextureType {
  SOLID,
  NOISE,
  CHECKER,
};

struct SolidTexture {
  float3 solid_color;
};

struct CheckerTexture {
  float3 odd_color;
  float3 even_color;
};

struct Texture {
  TextureType type;
  SolidTexture solid;
  CheckerTexture checker;
};

float3 solid_texture_color(const Texture *texture, float u, float v,
                           const float3 &p) {
  return texture->solid.solid_color;
}

float3 checker_texture_color(const Texture *texture, float u, float v,
                             const float3 &p) {
  const float sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
  if (sines < 0) {
    return texture->checker.odd_color;
  }
  return texture->checker.even_color;
}

float3 texture_color(const Texture *texture, float u, float v,
                     const float3 &p) {
  switch (texture->type) {
  case SOLID:
    return solid_texture_color(texture, u, v, p);
  case CHECKER:
    return checker_texture_color(texture, u, v, p);
  }
  const uniform float3 zeros = {0, 0, 0};
  return zeros;
}

static inline void set_sphere_uv(const float3 &p, float &u, float &v) {
  const float theta = acos(-p.y);
  const float phi = atan2(-p.z, p.x) + pi;
  u = phi / (2 * pi);
  v = theta / pi;
}

static inline void set_face_normal(const Ray &r, const float3 &outward_normal,
                                   HitInfo &hi) {
  hi.front_face = dot(r.dir, outward_normal) < 0;
  hi.normal = hi.front_face ? outward_normal : -outward_normal;
}

bool hit_sphere(const uniform Hittable &h, uniform float t_min, float t_max,
                const Ray &r, HitInfo &hi) {
  const uniform Sphere &s = h.sphere;
  float3 oc = r.orig - s.center;
  const float a = dot(r.dir, r.dir);
  const float hb = dot(oc, r.dir);
  const float c = dot(oc, oc) - s.radius * s.radius;
  const float discriminant = hb * hb - a * c;
  if (discriminant < 0) {
    return false;
  }

  float sqrtd = sqrt(discriminant);
  float root = (-hb - sqrtd) / a;
  if (root < t_min || t_max < root) {
    root = (-hb + sqrtd) / a;
    if (root < t_min || t_max < root) {
      return false;
    }
  }

  hi.t = root;
  hi.p = at(r, root);
  float3 outward_normal = (hi.p - s.center) / s.radius;
  set_face_normal(r, outward_normal, hi);
  set_sphere_uv(outward_normal, hi.u, hi.v);
  hi.material_id = h.material_id;
  return true;
}

static inline float3 moving_sphere_center(const uniform MovingSphere &ms,
                                          float time) {
  return ms.center0 + ((time - ms.time0) / (ms.time1 - ms.time0)) *
                          (ms.center1 - ms.center0);
}

bool hit_moving_sphere(const uniform Hittable &h, uniform float t_min,
                       float t_max, const Ray &r, HitInfo &hi) {
  const uniform MovingSphere &s = h.moving_sphere;
  const float3 center = moving_sphere_center(s, r.time);
  float3 oc = r.orig - center;
  const float a = dot(r.dir, r.dir);
  const float hb = dot(oc, r.dir);
  const float c = dot(oc, oc) - s.radius * s.radius;
  const float discriminant = hb * hb - a * c;
  if (discriminant < 0) {
    return false;
  }

  float sqrtd = sqrt(discriminant);
  float root = (-hb - sqrtd) / a;
  if (root < t_min || t_max < root) {
    root = (-hb + sqrtd) / a;
    if (root < t_min || t_max < root) {
      return false;
    }
  }

  hi.t = root;
  hi.p = at(r, root);
  float3 outward_normal = (hi.p - center) / s.radius;
  set_face_normal(r, outward_normal, hi);
  set_sphere_uv(outward_normal, hi.u, hi.v);
  hi.material_id = h.material_id;
  return true;
}

bool hit_xy_rect(const uniform XYRect &xy_rect, int material_id,
                 uniform float t_min, float t_max, const Ray &r, HitInfo &hi) {
  const float t = (xy_rect.k - r.orig.z) / r.dir.z;
  if (t < t_min || t > t_max) {
    return false;
  }

  const float x = r.orig.x + t * r.dir.x;
  const float y = r.orig.y + t * r.dir.y;
  if (x < xy_rect.x0 || x > xy_rect.x1 || y < xy_rect.y0 || y > xy_rect.y1) {
    return false;
  }

  hi.u = (x - xy_rect.x0) / (xy_rect.x1 - xy_rect.x0);
  hi.u = (y - xy_rect.y0) / (xy_rect.y1 - xy_rect.y0);
  hi.t = t;
  float3 outward_normal = {0, 0, 1};
  set_face_normal(r, outward_normal, hi);
  hi.material_id = material_id;
  hi.p = at(r, t);
  return true;
}

bool hit_xz_rect(const uniform XZRect &xz_rect, int material_id,
                 uniform float t_min, float t_max, const Ray &r, HitInfo &hi) {
  const float t = (xz_rect.k - r.orig.y) / r.dir.y;
  if (t < t_min || t > t_max) {
    return false;
  }

  const float x = r.orig.x + t * r.dir.x;
  const float z = r.orig.z + t * r.dir.z;
  if (x < xz_rect.x0 || x > xz_rect.x1 || z < xz_rect.z0 || z > xz_rect.z1) {
    return false;
  }

  hi.u = (x - xz_rect.x0) / (xz_rect.x1 - xz_rect.x0);
  hi.u = (z - xz_rect.z0) / (xz_rect.z1 - xz_rect.z0);
  hi.t = t;
  float3 outward_normal = {0, 1, 0};
  set_face_normal(r, outward_normal, hi);
  hi.material_id = material_id;
  hi.p = at(r, t);
  return true;
}

bool hit_yz_rect(const uniform YZRect &yz_rect, int material_id,
                 uniform float t_min, float t_max, const Ray &r, HitInfo &hi) {
  const float t = (yz_rect.k - r.orig.x) / r.dir.x;
  if (t < t_min || t > t_max) {
    return false;
  }

  const float y = r.orig.y + t * r.dir.y;
  const float z = r.orig.z + t * r.dir.z;
  if (y < yz_rect.y0 || y > yz_rect.y1 || z < yz_rect.z0 || z > yz_rect.z1) {
    return false;
  }

  hi.u = (y - yz_rect.y0) / (yz_rect.y1 - yz_rect.y0);
  hi.u = (z - yz_rect.z0) / (yz_rect.z1 - yz_rect.z0);
  hi.t = t;
  float3 outward_normal = {1, 0, 0};
  set_face_normal(r, outward_normal, hi);
  hi.material_id = material_id;
  hi.p = at(r, t);
  return true;
}

bool hit_box(const uniform Hittable &h, uniform float t_min, float t_max,
             const Ray &r, HitInfo &hi) {
  bool hit = false;
  float closest = t_max;

  if (hit_xy_rect(h.box.s0, h.material_id, t_min, closest, r, hi)) {
    closest = hi.t;
    hit = true;
  }
  if (hit_xy_rect(h.box.s1, h.material_id, t_min, closest, r, hi)) {
    closest = hi.t;
    hit = true;
  }
  if (hit_xz_rect(h.box.s2, h.material_id, t_min, closest, r, hi)) {
    closest = hi.t;
    hit = true;
  }
  if (hit_xz_rect(h.box.s3, h.material_id, t_min, closest, r, hi)) {
    closest = hi.t;
    hit = true;
  }
  if (hit_yz_rect(h.box.s4, h.material_id, t_min, closest, r, hi)) {
    closest = hi.t;
    hit = true;
  }
  if (hit_yz_rect(h.box.s5, h.material_id, t_min, closest, r, hi)) {
    closest = hi.t;
    hit = true;
  }
  return hit;
}

bool hit_obj(const uniform Hittable hlist[], const uniform Hittable &h,
             uniform float t_min, float t_max, const Ray &r, HitInfo &hi) {
  if (h.type == SPHERE) {
    return hit_sphere(h, t_min, t_max, r, hi);
  }
  if (h.type == MOVING_SPHERE) {
    return hit_moving_sphere(h, t_min, t_max, r, hi);
  }
  if (h.type == XY_RECT) {
    return hit_xy_rect(h.xy_rect, h.material_id, t_min, t_max, r, hi);
  }
  if (h.type == XZ_RECT) {
    return hit_xz_rect(h.xz_rect, h.material_id, t_min, t_max, r, hi);
  }
  if (h.type == YZ_RECT) {
    return hit_yz_rect(h.yz_rect, h.material_id, t_min, t_max, r, hi);
  }
  if (h.type == BOX) {
    return hit_box(h, t_min, t_max, r, hi);
  }
  return false;
}

HitInfo hit_any(const uniform Hittable hittable[], uniform int size,
                uniform float t_min, uniform float t_max, const Ray &r) {
  HitInfo hi;
  hi.hit = false;
  float closest = t_max;

  for (uniform int i = 0; i < size; ++i) {
    if (hit_obj(hittable, hittable[i], t_min, closest, r, hi)) {
      closest = hi.t;
      hi.hit = true;
    }
  }
  return hi;
}

struct AABB {
  float3 min;
  float3 max;
};

bool AABBHit(const uniform AABB &box, const Ray &r, float t_min, float t_max) {
  float t0 = t_min;
  float t1 = t_max;

  float3 tNear = (box.min - r.orig) / r.dir;
  float3 tFar = (box.max - r.orig) / r.dir;
  if (tNear.x > tFar.x) {
    float tmp = tNear.x;
    tNear.x = tFar.x;
    tFar.x = tmp;
  }
  t0 = max(tNear.x, t0);
  t1 = min(tFar.x, t1);

  if (tNear.y > tFar.y) {
    float tmp = tNear.y;
    tNear.y = tFar.y;
    tFar.y = tmp;
  }
  t0 = max(tNear.y, t0);
  t1 = min(tFar.y, t1);

  if (tNear.z > tFar.z) {
    float tmp = tNear.z;
    tNear.z = tFar.z;
    tFar.z = tmp;
  }
  t0 = max(tNear.z, t0);
  t1 = min(tFar.z, t1);

  return (t0 <= t1);
}

struct BVH2 {
  AABB bounds;
  uint8 prim_mask;
  int children[2];
};

bool hit_bvh(const uniform BVH2 bvh[], const uniform BVH2 &node,
             const uniform Hittable hlist[], uniform float t_min, float t_max,
             const Ray &r, HitInfo &hi) {

  if (!AABBHit(node.bounds, r, t_min, t_max)) {
    return false;
  }
  // bool hit_left;
  // if ((node.prim_mask & 0x01) != 0) {
  //   hit_left = hit_obj(hlist, hlist[node.children[0]], t_min, t_max, r, hi);
  // } else {
  //   hit_left = hit_bvh(bvh, bvh[node.children[0]], hlist, t_min, t_max, r,
  //   hi);
  // }
  // bool hit_right;
  // if ((node.prim_mask & 0x02) != 0) {
  //   hit_right = hit_obj(hlist, hlist[node.children[1]], t_min, hit_left ?
  //   hi.t : t_max, r, hi);
  // } else {
  //   hit_right = hit_bvh(bvh, bvh[node.children[1]], hlist, t_min, hit_left ?
  //   hi.t : t_max, r, hi);
  // }
  // return hit_left || hit_right;
  if (node.prim_mask != 0) {
    bool hit_left =
        hit_obj(hlist, hlist[node.children[0]], t_min, t_max, r, hi);
    bool hit_right = hit_obj(hlist, hlist[node.children[1]], t_min,
                             hit_left ? hi.t : t_max, r, hi);
    return hit_left || hit_right;
  }
  bool hit_left =
      hit_bvh(bvh, bvh[node.children[0]], hlist, t_min, t_max, r, hi);
  bool hit_right = hit_bvh(bvh, bvh[node.children[1]], hlist, t_min,
                           hit_left ? hi.t : t_max, r, hi);
  return hit_left || hit_right;
}

enum MaterialType {
  LAMBERTIAN,
  METALLIC,
  DIELECTRIC,
  DIFFUSE_LIGHT,
};

struct Material {
  MaterialType type;
  int albedo_texture_id;
  int emit_texture_id;
  float metal_fuzz;
  float dielectric_refraction_index;
};

struct ScatteredRay {
  bool was_scattered;
  float3 attenuation;
  Ray scattered;
};

ScatteredRay scatter_lambertian(const uniform Texture textures[],
                                const Material &material, const Ray &r,
                                varying RNGState *uniform rng,
                                const HitInfo &hit) {
  float3 scatter_dir = hit.normal + random_unit_vec(rng);
  if (near_zero(scatter_dir)) {
    scatter_dir = hit.normal;
  }

  ScatteredRay sr;
  sr.was_scattered = true;
  sr.scattered.orig = hit.p;
  sr.scattered.dir = scatter_dir;
  sr.scattered.time = r.time;
  sr.attenuation =
      texture_color(&textures[material.albedo_texture_id], hit.u, hit.v, hit.p);
  return sr;
}

static inline float3 reflect(const float3 &v, const float3 &n) {
  return v - 2 * dot(v, n) * n;
}

ScatteredRay scatter_metallic(const uniform Texture textures[],
                              const Material &material, const Ray &r,
                              varying RNGState *uniform rng,
                              const HitInfo &hit) {
  const float3 reflected = reflect(unit(r.dir), hit.normal);

  ScatteredRay sr;
  sr.scattered.orig = hit.p;
  sr.scattered.dir =
      reflected + material.metal_fuzz * random_in_unit_sphere(rng);
  sr.scattered.time = r.time;
  sr.attenuation =
      texture_color(&textures[material.albedo_texture_id], hit.u, hit.v, hit.p);
  sr.was_scattered = dot(reflected, hit.normal) > 0.0;
  return sr;
}

// schlick's approximation
static inline float reflectance(float cosine, float ref_idx) {
  float r0 = (1 - ref_idx) / (1 + ref_idx);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow((1 - cosine), 5);
}

static inline float3 refract(float3 uv, float3 n, float eta_ratio) {
  const float cos_theta = min(dot(-uv, n), 1.0);
  float3 perp = eta_ratio * (uv + cos_theta * n);
  float3 parallel = -sqrt(abs(1.0 - dot(perp, perp))) * n;
  return perp + parallel;
}

ScatteredRay scatter_dielectric(const Material &material, const Ray &r,
                                varying RNGState *uniform rng,
                                const HitInfo &hit) {
  ScatteredRay sr;
  const uniform float3 ones = {1.0, 1.0, 1.0};
  sr.attenuation = ones;

  const float ir = material.dielectric_refraction_index;
  const float refraction_ratio = hit.front_face ? 1.0 / ir : ir;

  const float3 unit_dir = unit(r.dir);
  const float cos_theta = min(dot(-unit_dir, hit.normal), 1.0);
  const float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
  const bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
  float3 dir;
  if (cannot_refract ||
      (reflectance(cos_theta, refraction_ratio) > frandom(rng))) {
    dir = reflect(unit_dir, hit.normal);
  } else {
    dir = refract(unit_dir, hit.normal, refraction_ratio);
  }

  sr.scattered.orig = hit.p;
  sr.scattered.dir = dir;
  sr.scattered.time = r.time;
  sr.was_scattered = true;
  return sr;
}

ScatteredRay scatter(const uniform Texture textures[], const Material &material,
                     const Ray &r, varying RNGState *uniform rng,
                     const HitInfo &hit) {
  if (material.type == LAMBERTIAN) {
    return scatter_lambertian(textures, material, r, rng, hit);
  }
  if (material.type == METALLIC) {
    return scatter_metallic(textures, material, r, rng, hit);
  }
  if (material.type == DIELECTRIC) {
    return scatter_dielectric(material, r, rng, hit);
  }
  ScatteredRay sr;
  sr.was_scattered = false;
  return sr;
}

color emitted(const uniform Texture textures[], const Material &material,
              const Ray &r, varying RNGState *uniform rng, const HitInfo &hit) {
  if (material.type == DIFFUSE_LIGHT) {
    return texture_color(&textures[material.emit_texture_id], hit.u, hit.v,
                         hit.p);
  }

  // By default emit nothing
  const uniform float3 zeros = {0, 0, 0};
  return zeros;
}

color ray_color(const uniform color background, const uniform BVH2 bvh[],
                uniform int bvh_root, const uniform Hittable hittable[],
                uniform int size, const uniform Material material[],
                const uniform Texture textures[], varying RNGState *uniform rng,
                const Ray &r, int depth) {
  const uniform int iflt_max = 0x7f800000; // +infinity
  const uniform float inf = floatbits(iflt_max);

  if (depth <= 0) {
    const uniform float3 zeros = {0, 0, 0};
    return zeros;
  }

  // bool hit_bvh(const uniform BVH2 bvh[],
  //              const uniform BVH2& node,
  //              const uniform Hittable hlist[],
  //              uniform float t_min, float t_max,
  //              const Ray& r, HitInfo& hi) {
  HitInfo hi;
  hi.hit = hit_bvh(bvh, bvh[bvh_root], hittable, 0.001, inf, r, hi);
  // HitInfo hi = hit_any(hittable, size, 0.001, inf, r);
  if (!hi.hit) {
    return background;
  }

  const Material m = material[hi.material_id];
  ScatteredRay sr = scatter(textures, m, r, rng, hi);
  const color emitted = emitted(textures, m, r, rng, hi);
  if (!sr.was_scattered) {
    return emitted;
  }

  return emitted + sr.attenuation * ray_color(background, bvh, bvh_root,
                                              hittable, size, material,
                                              textures, rng, sr.scattered,
                                              depth - 1);

  // float3 target = hi.p + hi.normal + random_unit_vec(rng);
  // Ray reflected;
  // reflected.orig = hi.p;
  // reflected.dir = target - hi.p;
  // return 0.5*ray_color(hittable, size, material, rng, reflected, depth-1);

  // float3 ud = unit(r.dir);
  // float3 t = 0.5*(ud.y + 1.0);
  // color a = {1.0, 1.0, 1.0};
  // color b = {0.5, 0.7, 1.0};
  // return a*(1.0-t) + b*t;
}

struct RenderParams {
  float3 look_from;
  float3 look_at;
  float3 vup;

  // const uniform float dist_to_focus = len(look_from - look_at);
  // const uniform float aperture = 2.0;
  float aperture;
  float vfov;

  float3 background;
  int img_w;
  int img_h;
  int msaa_samples;
  float time0;
  float time1;
  const Hittable *hittable;
  int size;
  const Material *material;
  const Texture *textures;
  const BVH2 *bvh;
  int bvh_root;
  uint8 *rbuf;
  uint8 *gbuf;
  uint8 *bbuf;
};

void render_tile(uniform int xs, uniform int xe, uniform int ys, uniform int ye,
                 uniform RenderParams &params, const uniform Camera &camera) {
  const uniform int max_depth = 50;
  foreach_tiled(y = ys... ye, x = xs... xe) {
    RNGState rng;
    seed_rng(&rng, make_seed(y, x));

    float3 c = {0, 0, 0};
    for (uniform int sample = 0; sample < params.msaa_samples; ++sample) {
      float u = ((float)x + frandom(&rng)) / (params.img_w - 1);
      float v = ((float)y + frandom(&rng)) / (params.img_h - 1);
      Ray r = get_ray(camera, &rng, u, v);
      c += ray_color(params.background, params.bvh, params.bvh_root,
                     params.hittable, params.size, params.material,
                     params.textures, &rng, r, max_depth);
    }

    // gamma correct
    const uniform float sample_scale = 1.0f / params.msaa_samples;
    float3 gcc = {sqrt(sample_scale * c.x), sqrt(sample_scale * c.y),
                  sqrt(sample_scale * c.z)};

    const int idx = y * params.img_w + x;
    const uniform float px_scale = 256.0f;
    // const color c = ray_color(hittable, size, r);
    params.rbuf[idx] = (uint8)(px_scale * clamp(gcc.x, 0.0f, 0.999));
    params.gbuf[idx] = (uint8)(px_scale * clamp(gcc.y, 0.0f, 0.999));
    params.bbuf[idx] = (uint8)(px_scale * clamp(gcc.z, 0.0f, 0.999));
  }
}

task void render_task(uniform RenderParams &params,
                      const uniform Camera &camera) {
  uniform int dx = 32;
  uniform int dy = 32;
  uniform int x_chunks = (params.img_w + (dx - 1)) / dx;
  uniform int x0 = (taskIndex % x_chunks) * dx;
  uniform int x1 = min(x0 + dx, params.img_w);
  uniform int y0 = (taskIndex / x_chunks) * dy;
  uniform int y1 = min(y0 + dy, params.img_h);
  render_tile(x0, x1, y0, y1, params, camera);
}

export void render_parallel(uniform RenderParams &params) {
  uniform float ratio = ((float)params.img_w) / params.img_h;
  const uniform float dist_to_focus = len(params.look_from - params.look_at);

  uniform Camera camera = MakeCamera(
      params.look_from, params.look_at, params.vup, params.vfov, ratio,
      params.aperture, dist_to_focus, params.time0, params.time1);
  uniform int dx = 32;
  uniform int dy = 32;
  uniform int x_chunks = (params.img_w + (dx - 1)) / dx;
  uniform int y_chunks = (params.img_h + (dy - 1)) / dy;
  uniform int chunks = x_chunks * y_chunks;
  launch[chunks] render_task(params, camera);
}
