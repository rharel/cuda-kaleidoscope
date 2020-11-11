#include <ctime>
#include <iomanip>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdexcept>

#include <curand_kernel.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "raytracer.h"

__host__ __device__ bool equals_approx(
    const float a,
    const float b,
    const float epsilon = 1e-6)
{
    return fabsf(a - b) < epsilon;
}

float radians(const float degrees)
{
    return degrees * float(M_PI) / 180;
}

#define VEC2_BINARY_OPERATOR(op) \
    template <typename T> \
    __host__ __device__ Vec2<T> operator ## op(const Vec2<T>& v, const T scalar) \
    { \
        return Vec2<T>( \
            v.x op scalar, \
            v.y op scalar\
        ); \
    } \
    \
    template <typename T> \
    __host__ __device__ Vec2<T> operator ## op(const T scalar, const Vec2<T>& v) \
    { \
        return Vec2<T>( \
            scalar op v.x, \
            scalar op v.y \
        ); \
    } \
    \
    template <typename T> \
    __host__ __device__ Vec2<T> operator ## op(const Vec2<T>& v0, const Vec2<T>& v1) \
    { \
        return Vec2<T>( \
            v0.x op v1.x, \
            v0.y op v1.y \
        ); \
    }

#define VEC2_BINARY_ASSIGNMENT_OPERATOR(op) \
    __host__ __device__ Vec2& operator ## op ## =(const T scalar) \
    { \
        return *this = *this op scalar; \
    } \
    \
    __host__ __device__ Vec2& operator ## op ## =(const Vec2& other) \
    { \
        return *this = *this op other; \
    }

#define VEC2_BINARY_METHOD(name, func) \
    __host__ __device__ Vec2 name(const T scalar) const \
    { \
        return Vec2( \
            func(x, scalar), \
            func(y, scalar) \
        ); \
    } \
    \
    __host__ __device__ Vec2 name(const Vec2& other) const \
    { \
        return Vec2( \
            func(x, other.x), \
            func(y, other.y) \
        ); \
    }

#define VEC2_UNARY_METHOD(name, func) \
    __host__ __device__ Vec2 name() const \
    { \
        return Vec2( \
            func(x), \
            func(y) \
        ); \
    }

template <typename T>
struct Vec2
{
    T x;
    T y;

    __host__ __device__ Vec2(T x, T y) : x(x), y(y) {}

    template <typename T2>
    __host__ __device__ Vec2<T2> as() const
    {
        return Vec2<T2>(x, y);
    }

    VEC2_BINARY_ASSIGNMENT_OPERATOR(-);
    VEC2_BINARY_ASSIGNMENT_OPERATOR(+);
    VEC2_BINARY_ASSIGNMENT_OPERATOR(/);
    VEC2_BINARY_ASSIGNMENT_OPERATOR(*);

    VEC2_BINARY_METHOD(mod, fmod);
    VEC2_UNARY_METHOD(abs, fabsf);
};

VEC2_BINARY_OPERATOR(-);
VEC2_BINARY_OPERATOR(+);
VEC2_BINARY_OPERATOR(/);
VEC2_BINARY_OPERATOR(*);

using Vec2f = Vec2<float>;
using Vec2z = Vec2<size_t>;

#define VEC3_BINARY_OPERATOR(op) \
    template <typename T> \
    __host__ __device__ Vec3<T> operator ## op(const Vec3<T>& v, const T scalar) \
    { \
        return Vec3<T>( \
            v.x op scalar, \
            v.y op scalar, \
            v.z op scalar \
        ); \
    } \
    \
    template <typename T> \
    __host__ __device__ Vec3<T> operator ## op(const T scalar, const Vec3<T>& v) \
    { \
        return Vec3<T>( \
            scalar op v.x, \
            scalar op v.y, \
            scalar op v.z \
        ); \
    } \
    \
    template <typename T> \
    __host__ __device__ Vec3<T> operator ## op(const Vec3<T>& v0, const Vec3<T>& v1) \
    { \
        return Vec3<T>( \
            v0.x op v1.x, \
            v0.y op v1.y, \
            v0.z op v1.z \
        ); \
    }

#define VEC3_BINARY_ASSIGNMENT_OPERATOR(op) \
    __host__ __device__ Vec3& operator ## op ## =(const T scalar) \
    { \
        return *this = *this op scalar; \
    } \
    \
    __host__ __device__ Vec3& operator ## op ## =(const Vec3& other) \
    { \
        return *this = *this op other; \
    }

#define VEC3_BINARY_METHOD(name, func) \
    __host__ __device__ Vec3 name(const T scalar) const \
    { \
        return Vec3( \
            func(x, scalar), \
            func(y, scalar), \
            func(z, scalar) \
        ); \
    } \
    \
    __host__ __device__ Vec3 name(const Vec3& other) const \
    { \
        return Vec3( \
            func(x, other.x), \
            func(y, other.y), \
            func(z, other.z) \
        ); \
    }

#define VEC3_UNARY_METHOD(name, func) \
    __host__ __device__ Vec3 name() const \
    { \
        return Vec3( \
            func(x), \
            func(y), \
            func(z) \
        ); \
    }

template <typename T>
struct Vec3
{
    T x;
    T y;
    T z;

    __host__ __device__ static Vec3 zero()
    {
        return Vec3<T>(0, 0, 0);
    }

    __host__ __device__ static Vec3 unit_x()
    {
        return Vec3<T>(1, 0, 0);
    }

    __host__ __device__ static Vec3 unit_y()
    {
        return Vec3<T>(0, 1, 0);
    }

    __host__ __device__ static Vec3 unit_z()
    {
        return Vec3<T>(0, 0, 1);
    }

    __host__ __device__ static Vec3 from_to(const Vec3& from, const Vec3& to)
    {
        return to - from;
    }

    __host__ __device__ Vec3(T x, T y, T z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3(const Vec2<T>& vector, T z) : x(vector.x), y(vector.y), z(z) {}

    __host__ __device__ float norm2() const
    {
        return x * x + y * y + z * z;
    }

    __host__ __device__ float norm() const
    {
        return sqrtf(norm2());
    }

    __host__ __device__ Vec3 cross(const Vec3& other) const
    {
        return Vec3f(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    __host__ __device__ float dot(const Vec3& other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ Vec3 unit() const
    {
        return *this / norm();
    }

    template <typename T2>
    __host__ __device__ Vec3<T2> as() const
    {
        return Vec3<T2>(x, y, z);
    }

    Vec3 operator-() const
    {
        return Vec3<T>(-x, -y, -z);
    }

    VEC3_BINARY_ASSIGNMENT_OPERATOR(-);
    VEC3_BINARY_ASSIGNMENT_OPERATOR(+);
    VEC3_BINARY_ASSIGNMENT_OPERATOR(/);
    VEC3_BINARY_ASSIGNMENT_OPERATOR(*);

    VEC3_BINARY_METHOD(min, fminf);
    VEC3_BINARY_METHOD(max, fmaxf);

    VEC3_UNARY_METHOD(round, roundf);
};

VEC3_BINARY_OPERATOR(-);
VEC3_BINARY_OPERATOR(+);
VEC3_BINARY_OPERATOR(/);
VEC3_BINARY_OPERATOR(*);

using Vec3f = Vec3<float>;
using Vec3u8 = Vec3<uint8_t>;

struct Quat
{
    float w;
    float x;
    float y;
    float z;

    __host__ __device__ static Quat from_angle_axis(
        const float angle,
        const Vec3f& axis)
    {
        return Quat(
            cosf(angle / 2),
            axis.x * sinf(angle / 2),
            axis.y * sinf(angle / 2),
            axis.z * sinf(angle / 2)
        );
    }

    __host__ __device__ Quat(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}

    __host__ __device__ float norm() const
    {
        return sqrtf(w * w + x * x + y * y + z * z);
    }

    __host__ __device__ Quat unit() const
    {
        const float d = norm();
        return Quat(w / d, x / d, y / d, z / d);
    }

    __host__ __device__ Quat operator*(const Quat& other) const
    {
        return Quat(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w - y * other.z + z * other.y,
            w * other.y + x * other.z + y * other.w - z * other.x,
            w * other.z - x * other.y + y * other.x + z * other.w
        );
    }

    __host__ __device__ Vec2f operator*(const Vec2f& v) const
    {
        const Vec3f u = *this * Vec3f(v.x, v.y, 0);
        return Vec2f(u.x, u.y);
    }

    __host__ __device__ Vec3f operator*(const Vec3f& v) const
    {
        const Quat a(0, v.x, v.y, v.z);
        const Quat b(w, -x, -y, -z);
        const Quat c = (*this * a) * b;
        return Vec3f(c.x, c.y, c.z);
    }
};

struct Plane
{
    Vec3f pivot;
    Vec3f normal;

    Plane(const Vec3f& pivot, const Vec3f& normal) : pivot(pivot), normal(normal)
    {
        CUDU_DEBUG_ASSERT(normal.norm2() == 1);
    }

    __host__ __device__ Vec2f in_local_space(const Vec3f& p) const
    {
        const Vec3f u = Vec3f(normal.y, -normal.x, 0).unit();
        const Vec3f v = normal.cross(u);
        return Vec2f(u.dot(p), v.dot(p));
    }
};

struct Background
{
    Plane plane;
    cudu::device::ConstArrayRef3D<float> texture;
    float texture_scale;
    Vec2f texture_offset;
    float texture_rotation_radians;

    Background(
        const Plane& plane,
        const cudu::device::ConstArrayRef3D<float> texture,
        const float texture_scale,
        const Vec2f& texture_offset,
        const float texture_rotation_radians) :
        plane(plane),
        texture(texture),
        texture_scale(texture_scale),
        texture_offset(texture_offset),
        texture_rotation_radians(texture_rotation_radians),
        texture_rotation(
            Quat::from_angle_axis(texture_rotation_radians, Vec3f::unit_z())
        ),
        texture_size(texture.shape()[0], texture.shape()[1]),
        texture_offset_times_scale(texture_offset * texture_scale),
        texture_scale_over_size(texture_scale / texture_size)
    {
        CUDU_DEBUG_ASSERT(texture_scale > 0);
    }

    __device__ Vec3f texel_color(const Vec3f& p) const
    {
        const Vec2f contact_plane = texture_rotation * plane.in_local_space(p);
        const Vec2f texel = (
            (contact_plane - texture_offset_times_scale).mod(texture_scale) /
            texture_scale_over_size + texture_size
        ).mod(texture_size);
        
        return Vec3f(
            texture(texel.x, texel.y, 0),
            texture(texel.x, texel.y, 1),
            texture(texel.x, texel.y, 2)
        );
    }

private:
    Quat texture_rotation;
    Vec2f texture_size;
    Vec2f texture_offset_times_scale;
    Vec2f texture_scale_over_size;
};

struct Scene
{
    Background background;
    float background_radiance;
    bool third_mirror_present;
    Plane mirrors[3];
    float mirror_albedo;
    float medium_scattering_coefficient;
    float medium_luminance_threshold;

    Scene(
        const cudu::device::ConstArrayRef3D<float> background_texture,
        const float background_texture_scale,
        const Vec2f& background_texture_offset,
        const float background_texture_rotation_radians,
        const float background_radiance,
        const bool third_mirror_present,
        const float mirror_internal_angle_radians,
        const float mirror_bend_angle_radians,
        const float mirror_albedo,
        const float medium_scattering_coefficient,
        const float medium_luminance_threshold):
        background(
            Plane(Vec3f::zero(), Vec3f::unit_y()),
            background_texture,
            background_texture_scale,
            background_texture_offset,
            background_texture_rotation_radians
        ),
        background_radiance(background_radiance),
        third_mirror_present(third_mirror_present),
        mirrors{
            Plane(
                -Vec3f::unit_z(),
                Quat::from_angle_axis(
                    +(float(M_PI) - mirror_internal_angle_radians) / 2,
                    Vec3f::unit_y()
                ) * 
                Quat::from_angle_axis(mirror_bend_angle_radians, Vec3f::unit_x()) *
                Vec3f::unit_z()
            ),
            Plane(
                -Vec3f::unit_z(),
                Quat::from_angle_axis(
                    -(float(M_PI) - mirror_internal_angle_radians) / 2, 
                    Vec3f::unit_y()
                ) * 
                Quat::from_angle_axis(mirror_bend_angle_radians, Vec3f::unit_x()) *
                Vec3f::unit_z()
            ),
            Plane(Vec3f::unit_z(), -Vec3f::unit_z())
        },
        mirror_albedo(mirror_albedo),
        medium_scattering_coefficient(medium_scattering_coefficient),
        medium_luminance_threshold(medium_luminance_threshold)
    {
    }
};

__device__ Vec2z unravel_index(
    const size_t index,
    const size_t extent)
{
    return Vec2z(
        index / extent,
        index % extent
    );
}

struct Camera
{
    Vec3f position;
    Vec3f direction;
    float field_of_view_radians;

    Camera(
        const Vec3f& position,
        const Vec3f& direction,
        const float field_of_view_radians) :
        position(position),
        direction(direction),
        field_of_view_radians(field_of_view_radians)
    {
        CUDU_DEBUG_ASSERT(equals_approx(direction.norm2(), 1));
        CUDU_DEBUG_ASSERT(field_of_view_radians > 0);
    }
};

struct Stratifier
{
    size_t resolution;
    size_t stratum_index = 0;

    Stratifier(const size_t resolution = 1) : resolution(resolution) {}

    __device__ Vec2f sample(curandStatePhilox4_32_10& rng)
    {
        const Vec2z stratum = unravel_index(stratum_index, resolution);
        const Vec2f random_offset(curand_uniform(&rng), curand_uniform(&rng));
        stratum_index = (stratum_index + 1) % (resolution * resolution);
        return (stratum.as<float>() + random_offset) / float(resolution);
    }
};

struct Ray
{
    Vec3f origin;
    Vec3f direction;

    __host__ __device__ Ray(
        const Vec3f& origin,
        const Vec3f& direction) :
        origin(origin),
        direction(direction)
    {
    }

    __host__ __device__ Vec3f at(const float t) const
    {
        return origin + direction * t;
    }
};

struct Trace
{
    Ray ray;
    Vec3f color = Vec3f(1, 1, 1);
    uint32_t depth = 0;
    bool complete = false;

    __host__ __device__ Trace() :
        ray(Vec3f::zero(), Vec3f::unit_y()),
        color(0, 0, 0),
        complete(true)
    {
    }

    __host__ __device__ Trace(const Ray& ray) : 
        ray(ray), 
        color(1, 1, 1), 
        complete(false)
    {
    }
};

__global__ void rng_state_setup_kernel(
    const unsigned long seed,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> state)
{
    if (CUDU_THREAD_ID() < state.size())
    {
        curand_init(seed, CUDU_THREAD_ID(), 0, &state[CUDU_THREAD_ID()]);
    }
}

__global__ void trace_init_kernel(
    const Camera camera,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::device::ArrayRef2D<Stratifier> stratifiers,
    cudu::device::ArrayRef2D<Trace> traces)
{
    if (CUDU_THREAD_ID() >= traces.size())
    {
        return;
    }

    const Vec2z pixel = unravel_index(CUDU_THREAD_ID(), traces.shape()[1]);
    
    if (!traces(pixel.x, pixel.y).complete)
    {
        return;
    }

    const Vec2f pixel_screen = (
        pixel.as<float>() + 
        stratifiers(pixel.x, pixel.y).sample(rng_state(pixel.x, pixel.y))
    );
    const Vec2f image_size_screen(traces.shape()[0], traces.shape()[1]);
    const Vec2f pixel_view_plane = pixel_screen / image_size_screen - 0.5f;
    const float image_size_ratio = image_size_screen.y / image_size_screen.x;
    const float image_width_world = 2 * tanf(camera.field_of_view_radians / 2);
    const float image_height_world = image_width_world * image_size_ratio;
    const Vec2f image_size_world(image_width_world, image_height_world);
    
    const Vec3f pixel_camera(pixel_view_plane * image_size_world, 1);
    const Vec3f camera_z = camera.direction;
    const Vec3f camera_x = camera_z.cross(Vec3f::unit_y()).unit();
    const Vec3f camera_y = camera_z.cross(camera_x);
    const float R[3][3] = {
        {camera_x.x, camera_y.x, camera_z.x},
        {camera_x.y, camera_y.y, camera_z.y},
        {camera_x.z, camera_y.z, camera_z.z}
    };
    const Vec3f pixel_world = camera.position + Vec3f(
        R[0][0] * pixel_camera.x + R[0][1] * pixel_camera.y + R[0][2] * pixel_camera.z, 
        R[1][0] * pixel_camera.x + R[1][1] * pixel_camera.y + R[1][2] * pixel_camera.z,
        R[2][0] * pixel_camera.x + R[2][1] * pixel_camera.y + R[2][2] * pixel_camera.z
    );
    traces(pixel.x, pixel.y) = Trace(Ray(
        camera.position, 
        Vec3f::from_to(camera.position, pixel_world).unit()
    ));
}

struct Intersection
{
    bool exists;
    float t;

    __host__ __device__ Intersection() : exists(false), t(-1) {}
    __host__ __device__ Intersection(const float t) : exists(t >= 0), t(t) {}
};

__device__ Intersection intersection_with_plane(
    const Ray& ray,
    const Plane& plane)
{
    const float denominator = plane.normal.dot(ray.direction);
    return (
        equals_approx(denominator, 0) ?
        Intersection() :
        Intersection(Vec3f::from_to(ray.origin, plane.pivot).dot(plane.normal) / denominator)
    );
}

__device__ Intersection intersection_with_medium(
    const float scattering_coefficient,
    curandStatePhilox4_32_10& rng)
{
    const float u = curand_uniform(&rng);
    const float t = log(1 / (1 - u)) / scattering_coefficient;
    return Intersection(t);
}

__device__ float perceived_luminance(const Vec3f& color_bgr)
{
    // https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
    return (
        0.0722f * color_bgr.x + 
        0.5870f * color_bgr.y + 
        0.2990f * color_bgr.z
    );
}

__global__ void trace_kernel(
    const Scene scene,
    const uint32_t max_trace_depth,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::device::ArrayRef2D<Trace> traces,
    cudu::device::ArrayRef3D<float> ray_color_sum,
    cudu::device::ArrayRef2D<unsigned> ray_count)
{
    if (CUDU_THREAD_ID() >= traces.size())
    {
        return;
    }

    const Vec2z pixel = unravel_index(CUDU_THREAD_ID(), traces.shape()[1]);
    Trace& trace = traces(pixel.x, pixel.y);

    if (trace.complete || trace.depth > max_trace_depth)
    {
        return;
    }

    const Intersection intersections[4] = {
        intersection_with_plane(trace.ray, scene.background.plane),
        intersection_with_plane(trace.ray, scene.mirrors[0]),
        intersection_with_plane(trace.ray, scene.mirrors[1]),
        scene.third_mirror_present ? intersection_with_plane(trace.ray, scene.mirrors[2]) : Intersection(),
    };

    constexpr size_t BACKGROUND = 0;
    constexpr size_t MIRROR_0 = 1;
    constexpr size_t MIRROR_1 = 2;
    constexpr size_t MIRROR_2 = 3;
    constexpr size_t NONE = 4;
    size_t contact_index = NONE;
    
    for (size_t i = 0; i < 4; ++i)
    {
        if (intersections[i].exists && (
            contact_index == NONE || 
            intersections[i].t < intersections[contact_index].t))
        {
            contact_index = i;
        }
    }

    if (contact_index == NONE)
    {
        trace.complete = true;
        ray_count(pixel.x, pixel.y) += 1;
        return;
    }

    const Intersection& intersection_plane = intersections[contact_index];
    const Intersection intersection_medium = intersection_with_medium(
        scene.medium_scattering_coefficient, 
        rng_state(pixel.x, pixel.y)
    );
    
    if (intersection_medium.t < intersection_plane.t)
    {
        const Vec3f contact_medium = trace.ray.at(intersection_medium.t);
        const Vec3f contact_ground(contact_medium.x, 0, contact_medium.z);
        const Vec3f texel_color = scene.background.texel_color(contact_ground);

        if (perceived_luminance(texel_color) >= scene.medium_luminance_threshold)
        {
            trace.ray.origin = contact_medium;
            trace.ray.direction = Vec3f(0, -1, 0);
            trace.depth += 1;
            return;
        }
    }
    
    const Vec3f contact_world = trace.ray.at(intersection_plane.t);

    if (contact_index == BACKGROUND)
    {
        const Vec3f texel_color = scene.background.texel_color(contact_world);
        const float cos_angle_of_incidence = fabsf(trace.ray.direction.dot(scene.background.plane.normal));
        
        trace.color *= cos_angle_of_incidence * (texel_color * scene.background_radiance).min(1);
        trace.complete = true;

        ray_color_sum(pixel.x, pixel.y, 0) += trace.color.x;
        ray_color_sum(pixel.x, pixel.y, 1) += trace.color.y;
        ray_color_sum(pixel.x, pixel.y, 2) += trace.color.z;

        ray_count(pixel.x, pixel.y) += 1;
    }
    else if (contact_index == MIRROR_0 || 
             contact_index == MIRROR_1 || 
             contact_index == MIRROR_2)
    {
        trace.color *= scene.mirror_albedo;
        
        constexpr float DARK_THRESHOLD = 1 / 255.f;

        if (trace.color.x < DARK_THRESHOLD &&
            trace.color.y < DARK_THRESHOLD && 
            trace.color.z < DARK_THRESHOLD)
        {
            trace.complete = true;
            ray_count(pixel.x, pixel.y) += 1;
        }
        else
        {
            // Compuate reflection:
            //
            //         n
            //    a--->|--->r
            //     \ b | b /
            //      \  |  /
            //       \ | /
            //        \|/
            // --------------------
            const Plane& mirror = scene.mirrors[contact_index - 1];
            const Vec3f a = Vec3f::from_to(contact_world, trace.ray.origin);
            const Vec3f n = mirror.normal * a.dot(mirror.normal);
            const Vec3f b = Vec3f::from_to(a, n);
            const Vec3f reflection = (a + 2.f * b).unit();

            // Pertrude origin along surface normal to avoid self-intersection 
            // due to rounding errors.
            trace.ray.origin = contact_world + 1e-5f * mirror.normal;  
            trace.ray.direction = reflection;
            trace.depth += 1;
        }
    }
}

__global__ void image_bgr_kernel(
    const cudu::device::ConstArrayRef3D<float> ray_color_sum,
    const cudu::device::ConstArrayRef2D<unsigned> ray_count,
    cudu::device::ArrayRef3D<unsigned char> image_bgr)
{
    if (CUDU_THREAD_ID() >= ray_count.size())
    {
        return;
    }

    const Vec2z pixel = unravel_index(CUDU_THREAD_ID(), image_bgr.shape()[1]);
    const Vec3f color_sum(
        ray_color_sum(pixel.x, pixel.y, 0),
        ray_color_sum(pixel.x, pixel.y, 1),
        ray_color_sum(pixel.x, pixel.y, 2)
    );
    const Vec3f color_f32 = color_sum / fmaxf(ray_count(pixel.x, pixel.y), 1);
    const Vec3u8 color_u8   = (255.f * color_f32).round().as<uint8_t>();

    image_bgr(pixel.y, pixel.x, 0) = color_u8.x;
    image_bgr(pixel.y, pixel.x, 1) = color_u8.y;
    image_bgr(pixel.y, pixel.x, 2) = color_u8.z;
}


cudu::host::Array3D<unsigned char> kaleido::raytraced_image(const Config& config)
{
    const cv::Mat texture_cv = cv::imread(
        config.background_texture_path,
        cv::ImreadModes::IMREAD_COLOR
    );
    if (texture_cv.dims == 0)
    {
        throw std::runtime_error("texture read failed");
    }
    cudu::host::Array3D<float> texture_h({
        size_t(texture_cv.cols),
        size_t(texture_cv.rows),
        size_t(texture_cv.channels())
    });
    for (size_t i = 0; i < texture_h.size(); ++i)
    {
        texture_h[i] = texture_cv.data[i] / 255.f;
    }
    cudu::device::Array3D<float> background_texture;
    background_texture.upload(texture_h);

    const Scene scene(
        background_texture,
        config.background_texture_scale,
        Vec2f(config.background_texture_offset[0], config.background_texture_offset[1]),
        radians(config.background_texture_rotation_degrees),
        config.background_radiance,
        config.three_mirrors,
        radians(config.mirror_internal_angle_degrees),
        radians(config.mirror_bend_angle_degrees),
        config.mirror_albedo,
        config.medium_scattering_coefficient,
        config.medium_luminance_threshold
    );
    const Vec3f camera_position = (
        Quat::from_angle_axis(radians(config.camera_position_azimuth_angle_degrees), Vec3f::unit_y()) *
        Quat::from_angle_axis(radians(config.camera_position_polar_angle_degrees), Vec3f::unit_x()) *
        Vec3f(0, 1, 0)).unit() * config.camera_position_radius;

    const Camera camera(
        camera_position,
        Vec3f::from_to(camera_position, Vec3f::zero()).unit(),
        radians(config.camera_field_of_view_degrees)
    );

    const cudu::Shape2D image_resolution(config.image_width, config.image_height);
    const cudu::Shape3D image_shape(config.image_width, config.image_height, 3);
    
    const auto workload = cudu::Workload::for_jobs(unsigned(image_resolution.size()));
    
    cudu::device::Array2D<Stratifier> stratifiers(image_resolution, Stratifier(config.stratifier_resolution));
    cudu::device::Array2D<curandStatePhilox4_32_10> rng(image_resolution);
    cudu::device::Array2D<Trace> traces(image_resolution, Trace());
    cudu::device::Array3D<float> ray_color_sum(image_shape, 0);
    cudu::device::Array2D<unsigned> ray_count(image_resolution, 0);
    
    CUDU_LAUNCH(rng_state_setup_kernel, workload, config.stratifier_rng_seed, rng);

    std::clock_t start_clock = std::clock();
    for (size_t i = 0; i < config.nr_render_iterations; ++i)
    {
        CUDU_LAUNCH(trace_init_kernel, workload, camera, rng, stratifiers, traces);
        CUDU_LAUNCH(trace_kernel, workload, scene, config.max_trace_depth, rng, traces, ray_color_sum, ray_count);
    }
    std::clock_t end_clock = std::clock();

    if (config.debug)
    {
        cudu::host::Array2D<unsigned> h_ray_count = ray_count.download();
        
        float ray_count_mean = 0;
        for (unsigned count : h_ray_count) { ray_count_mean += count; }
        ray_count_mean /= h_ray_count.size();
        
        float ray_count_std = 0;
        for (unsigned count : h_ray_count) { ray_count_std += abs(count - float(ray_count_mean)); }
        ray_count_std /= h_ray_count.size();
        
        const float cpu_s = float(end_clock - start_clock) / CLOCKS_PER_SEC;

        std::cout
            << std::fixed << std::setprecision(2)
            << "CPU time: " << cpu_s << " s\n"
            << "Rays/pixel mean: " << ray_count_mean << "\n"
            << "Rays/pixel std: " << ray_count_std << "\n"
            << "Rays/pixel/s mean: " << ray_count_mean / cpu_s << std::endl;
    }

    cudu::device::Array3D<unsigned char> image_bgr(image_shape, 0);
    CUDU_LAUNCH(image_bgr_kernel, workload, ray_color_sum, ray_count, image_bgr);
    
    return image_bgr.download();
}
