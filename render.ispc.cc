
typedef float<3> float3;

typedef float3 color;

static inline float random_range(varying RNGState *uniform rng, float fmin, float fmax) {
  return fmin + (fmax-fmin)*frandom(rng);
}

static inline float dot(float3 a, float3 b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline float len(float3 a) {
  return sqrt(dot(a,a));
}

static inline float3 unit(float3 a) {
  return a/len(a);
}

static inline bool near_zero(float3 a) {
  const float s = 1e-8;
  return (abs(a.x) < s) && (abs(a.y) < s) && (abs(a.z) < s);
}

static inline uniform float dot(uniform float3 a, uniform float3 b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline uniform float len(uniform float3 a) {
  return sqrt(dot(a,a));
}

static inline uniform float3 cross(uniform float3 a, uniform float3 b) {
  uniform float3 r;
  r.x = a.y*b.z - a.z*b.y;
  r.y = a.z*b.x - a.x*b.z;
  r.z = a.x*b.y - a.y*b.x;
  return r;
}

static inline uniform float3 unit(uniform float3 a) {
  return a/len(a);
}

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

static inline float3 random_vec(varying RNGState *uniform rng, float fmin, float fmax) {
  float3 v = {random_range(rng, fmin, fmax),
              random_range(rng, fmin, fmax),
              random_range(rng, fmin, fmax)};
  return v;
}

static inline float3 random_in_unit_sphere(varying RNGState *uniform rng) {
  while(true) {
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
  while(true) {
    float3 v = {random_range(rng, -1, 1),
                random_range(rng, -1, 1),
                0.0};
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

static inline float3 at(const Ray& r, float t) {
  return r.orig + t*r.dir;
}

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

uniform Camera MakeCamera(uniform float3 look_from,
                          uniform float3 look_at,
                          uniform float3 vup,
                          uniform float vfov,
                          uniform float aspect_ratio,
                          uniform float aperture,
                          uniform float focus_dist,
                          uniform float time0,
                          uniform float time1) {
  const uniform float pi = 0x1.921fb54442d18p+1;
  const uniform float theta = pi*vfov/180.0;
  const uniform float th = tan(theta/2);
  const uniform float height = 2.0*th;
  const uniform float width = aspect_ratio*height;

  const uniform float3 w = unit(look_from - look_at);
  const uniform float3 u = unit(cross(vup, w));
  const uniform float3 v = cross(w, u);

  uniform Camera camera;
  camera.w = w;
  camera.u = u;
  camera.v = v;
  camera.origin = look_from;
  camera.horizontal = focus_dist*width*u;
  camera.vertical = focus_dist*height*v;
  camera.lower_left =  camera.origin - camera.horizontal/2 - camera.vertical/2 - focus_dist*w;
  camera.lens_radius = aperture/2;
  camera.time0 = time0;
  camera.time1 = time1;
  return camera;
}

Ray get_ray(const uniform Camera& camera,
            varying RNGState *uniform rng,
            float s, float t) {
  const float3 rd = camera.lens_radius*random_in_unit_disk(rng);
  const float3 offset = camera.u*rd.x + camera.v*rd.y;

  Ray r;
  r.orig = camera.origin + offset;
  r.dir = camera.lower_left + s*camera.horizontal + t*camera.vertical - camera.origin - offset;
  r.time = random_range(rng, camera.time0, camera.time1);
  return r;
}

struct HitInfo {
  bool hit;
  float3 p;
  float3 normal;
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
};

struct Hittable {
  HittableType type;
  int material_id;
  Sphere sphere;
  MovingSphere moving_sphere;
};

static inline void set_face_normal(const Ray& r, const float3& outward_normal, HitInfo& hi) {
  hi.front_face = dot(r.dir, outward_normal) < 0;
  hi.normal = hi.front_face ? outward_normal : -outward_normal;
}

bool hit_sphere(const uniform Hittable& h,
                uniform float t_min, float t_max,
                const Ray& r, HitInfo& hi) {
  const uniform Sphere& s = h.sphere;
  float3 oc = r.orig - s.center;
  const float a = dot(r.dir, r.dir);
  const float hb = dot(oc, r.dir);
  const float c = dot(oc, oc) - s.radius*s.radius;
  const float discriminant = hb*hb - a*c;
  if (discriminant < 0) {
    return false;
  }

  float sqrtd = sqrt(discriminant);
  float root = (-hb - sqrtd)/a;
  if (root < t_min || t_max < root) {
    root = (-hb + sqrtd)/a;
    if (root < t_min || t_max < root) {
      return false;
    }
  }

  hi.t = root;
  hi.p = at(r, root);
  float3 outward_normal = (hi.p - s.center)/s.radius;
  set_face_normal(r, outward_normal, hi);
  hi.material_id = h.material_id;
  return true;
}

static inline float3 moving_sphere_center(const uniform MovingSphere& ms, float time) {
  return ms.center0 + ((time - ms.time0)/(ms.time1 - ms.time0))*(ms.center1 - ms.center0);
}

bool hit_moving_sphere(const uniform Hittable& h,
                uniform float t_min, float t_max,
                const Ray& r, HitInfo& hi) {
  const uniform MovingSphere& s = h.moving_sphere;
  const float3 center = moving_sphere_center(s, r.time);
  float3 oc = r.orig - center;
  const float a = dot(r.dir, r.dir);
  const float hb = dot(oc, r.dir);
  const float c = dot(oc, oc) - s.radius*s.radius;
  const float discriminant = hb*hb - a*c;
  if (discriminant < 0) {
    return false;
  }

  float sqrtd = sqrt(discriminant);
  float root = (-hb - sqrtd)/a;
  if (root < t_min || t_max < root) {
    root = (-hb + sqrtd)/a;
    if (root < t_min || t_max < root) {
      return false;
    }
  }

  hi.t = root;
  hi.p = at(r, root);
  float3 outward_normal = (hi.p - center)/s.radius;
  set_face_normal(r, outward_normal, hi);
  hi.material_id = h.material_id;
  return true;
}

bool hit_obj(const uniform Hittable& h,
             uniform float t_min, float t_max,
             const Ray& r, HitInfo& hi) {
  if (h.type == SPHERE) {
    return hit_sphere(h, t_min, t_max, r, hi);
  }
  if (h.type == MOVING_SPHERE) {
    return hit_moving_sphere(h, t_min, t_max, r, hi);
  }
  return false;
}

HitInfo hit_any(const uniform Hittable hittable[], uniform int size,
             uniform float t_min, uniform float t_max,
             const Ray& r) {
  HitInfo hi;
  hi.hit = false;
  float closest = t_max;

  for (uniform int i = 0; i < size; ++i) {
    if (hit_obj(hittable[i], t_min, closest, r, hi)) {
      closest = hi.t;
      hi.hit = true;
    }
  }
  return hi;
}

enum MaterialType {
  LAMBERTIAN,
  METALLIC,
  DIELECTRIC,
};

struct Material {
  MaterialType type;
  float3 albedo;
  float metal_fuzz;
  float dielectric_refraction_index;
};

struct ScatteredRay {
  bool was_scattered;
  float3 attenuation;
  Ray scattered;
};

ScatteredRay scatter_lambertian(const Material& material,
                                const Ray& r,
                                varying RNGState *uniform rng,
                                const HitInfo& hit) {
  float3 scatter_dir = hit.normal + random_unit_vec(rng);
  if (near_zero(scatter_dir)) {
    scatter_dir = hit.normal;
  }

  ScatteredRay sr;
  sr.was_scattered = true;
  sr.scattered.orig = hit.p;
  sr.scattered.dir = scatter_dir;
  sr.scattered.time = r.time;
  sr.attenuation = material.albedo;
  return sr;
}

static inline float3 reflect(const float3& v, const float3& n) {
  return v - 2*dot(v, n)*n;
}

ScatteredRay scatter_metallic(const Material& material,
                                const Ray& r,
                                varying RNGState *uniform rng,
                                const HitInfo& hit) {
  const float3 reflected = reflect(unit(r.dir), hit.normal);

  ScatteredRay sr;
  sr.scattered.orig = hit.p;
  sr.scattered.dir = reflected + material.metal_fuzz*random_in_unit_sphere(rng);
  sr.scattered.time = r.time;
  sr.attenuation = material.albedo;
  sr.was_scattered = dot(reflected, hit.normal) > 0.0;
  return sr;
}

// schlick's approximation
static inline float reflectance(float cosine, float ref_idx) {
  float r0 = (1 - ref_idx)/(1 + ref_idx);
  r0 = r0*r0;
  return r0 + (1 - r0)*pow((1 - cosine), 5);
}

static inline float3 refract(float3 uv, float3 n, float eta_ratio) {
  const float cos_theta = min(dot(-uv, n), 1.0);
  float3 perp = eta_ratio*(uv + cos_theta*n);
  float3 parallel = -sqrt(abs(1.0 - dot(perp, perp)))*n;
  return perp + parallel;
}

ScatteredRay scatter_dielectric(const Material& material,
                                const Ray& r,
                                varying RNGState *uniform rng,
                                const HitInfo& hit) {
  ScatteredRay sr;
  const uniform float3 ones = {1.0, 1.0, 1.0};
  sr.attenuation = ones;

  const float ir = material.dielectric_refraction_index;
  const float refraction_ratio = hit.front_face ? 1.0/ir : ir;

  const float3 unit_dir = unit(r.dir);
  const float cos_theta = min(dot(-unit_dir, hit.normal), 1.0);
  const float sin_theta = sqrt(1.0f - cos_theta*cos_theta);
  const bool cannot_refract = refraction_ratio*sin_theta > 1.0f;
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

ScatteredRay scatter(const Material& material,
                     const Ray& r,
                     varying RNGState *uniform rng,
                     const HitInfo& hit) {
  if (material.type == LAMBERTIAN) {
    return scatter_lambertian(material, r, rng, hit);
  }
  if (material.type == METALLIC) {
    return scatter_metallic(material, r, rng, hit);
  }
  if (material.type == DIELECTRIC) {
    return scatter_dielectric(material, r, rng, hit);
  }
  ScatteredRay sr;
  sr.was_scattered = false;
  return sr;
}

color ray_color(const uniform Hittable hittable[],
                uniform int size,
                const uniform Material material[],
                varying RNGState *uniform rng,
                const Ray& r, int depth) {
  const uniform int iflt_max = 0x7f800000; // +infinity
  const uniform float inf = floatbits(iflt_max);

  if (depth <= 0) {
    const uniform float3 zeros = {0, 0, 0};
    return zeros;
  }

  HitInfo hi = hit_any(hittable, size, 0.001, inf, r);
  if (hi.hit) {
    const Material m = material[hi.material_id];
    ScatteredRay sr = scatter(m, r, rng, hi);
    if (sr.was_scattered) {
      return sr.attenuation*ray_color(hittable, size, material, rng, sr.scattered, depth-1);
    }
    // float3 target = hi.p + hi.normal + random_unit_vec(rng);
    // Ray reflected;
    // reflected.orig = hi.p;
    // reflected.dir = target - hi.p;
    // return 0.5*ray_color(hittable, size, material, rng, reflected, depth-1);
    const uniform float3 zeros = {0, 0, 0};
    return zeros;
  }
  float3 ud = unit(r.dir);
  float3 t = 0.5*(ud.y + 1.0);
  color a = {1.0, 1.0, 1.0};
  color b = {0.5, 0.7, 1.0};
  return a*(1.0-t) + b*t;
}

void render_tile(uniform int xs, uniform int xe,
                 uniform int ys, uniform int ye,
                 uniform int img_w, uniform int img_h,
                 uniform int msaa_samples,
                 const uniform Camera& camera,
                 const uniform Hittable hittable[], uniform int size,
                 const uniform Material material[],
                 uniform uint8 rbuf[],
                 uniform uint8 gbuf[],
                 uniform uint8 bbuf[]) {
  const uniform int max_depth = 50;
  foreach_tiled(y = ys ... ye, x = xs ... xe) {
    RNGState rng;
    seed_rng(&rng, make_seed(y, x));

    float3 c = {0, 0, 0};
    for (uniform int sample = 0; sample < msaa_samples; ++sample) {
      float u = ((float)x + frandom(&rng))/(img_w-1);
      float v = ((float)y + frandom(&rng))/(img_h-1);
      Ray r = get_ray(camera, &rng, u, v);
      c += ray_color(hittable, size, material, &rng, r, max_depth);
    }

    // gamma correct
    const uniform float sample_scale = 1.0f/msaa_samples;
    float3 gcc = {sqrt(sample_scale*c.x),
                  sqrt(sample_scale*c.y),
                  sqrt(sample_scale*c.z)};

    const int idx = y*img_w + x;
    const uniform float px_scale = 256.0f;
    // const color c = ray_color(hittable, size, r);
    rbuf[idx] = (uint8)(px_scale*clamp(gcc.x, 0.0f, 0.999));
    gbuf[idx] = (uint8)(px_scale*clamp(gcc.y, 0.0f, 0.999));
    bbuf[idx] = (uint8)(px_scale*clamp(gcc.z, 0.0f, 0.999));
  }
}

task void render_task(
                 uniform int img_w, uniform int img_h,
                 uniform int msaa_samples,
                 const uniform Camera& camera,
                 const uniform Hittable hittable[], uniform int size,
                 const uniform Material material[],
                 uniform uint8 rbuf[],
                 uniform uint8 gbuf[],
                 uniform uint8 bbuf[]) {
  uniform int dx = 32;
  uniform int dy = 32;
  uniform int x_chunks = (img_w + (dx-1)) / dx;
  uniform int x0 = (taskIndex % x_chunks) * dx;
  uniform int x1 = min(x0 + dx, img_w);
  uniform int y0 = (taskIndex / x_chunks) * dy;
  uniform int y1 = min(y0 + dy, img_h);
  render_tile(x0, x1,
              y0, y1,
              img_w, img_h,
              msaa_samples,
              camera,
              hittable, size,
              material,
              rbuf, gbuf, bbuf);
}

export void render_parallel(const uniform Hittable hittable[], uniform int size,
                            const uniform Material material[],
                            uniform int img_w, uniform int img_h,
                            uniform float time0, uniform float time1,
                            uniform int msaa_samples,
                            uniform uint8 rbuf[],
                            uniform uint8 gbuf[],
                            uniform uint8 bbuf[]) {
  uniform float ratio = ((float)img_w)/img_h;
  const uniform float3 look_from = {13, 2, 3};
  const uniform float3 look_at = {0, 0, 0};
  const uniform float3 vup = {0, 1, 0};
  // const uniform float dist_to_focus = len(look_from - look_at);
  // const uniform float aperture = 2.0;
  const uniform float aperture = 0.1;
  const uniform float dist_to_focus = 10.0f;

  uniform Camera camera = MakeCamera(look_from, look_at, vup, 20, ratio, aperture, dist_to_focus, time0, time1);

  uniform int dx = 32;
  uniform int dy = 32;
  uniform int x_chunks = (img_w + (dx-1)) / dx;
  uniform int y_chunks = (img_h + (dy-1)) / dy;
  uniform int chunks = x_chunks * y_chunks;
  launch[chunks] render_task(img_w, img_h,
                             msaa_samples,
                             camera,
                             hittable, size,
                             material,
                             rbuf, gbuf, bbuf);
}
