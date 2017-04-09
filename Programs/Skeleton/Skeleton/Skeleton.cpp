//=============================================================================================
// Framework for the ray tracing homework
// ---------------------------------------------------------------------------------------------
// Name    : 
// Neptun : 
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

struct vec3 {
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + 0.000001));
	}
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}



void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates

	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(float f) {
		return vec4(v[0] * f, v[1] * f, v[2] * f);
	}

	vec4 operator+(vec4& v2) {
		return vec4(v[0] + v2.v[0], v[1] + v2.v[1], v[2] + v2.v[2]);
	}

	vec4 operator-(vec4& v2) {
		return vec4(v[0] - v2.v[0], v[1] - v2.v[1], v[2] - v2.v[2]);
	}

	vec4 operator/(float f) {
		return (*this)*(1/f);
	}

	// cross
	vec4 operator%(vec4& v2) {
		return vec4(v[1] * v2.v[2] - v[2] * v2.v[1], v[2] * v2.v[0] - v[0] * v2.v[2], v[0] * v2.v[1] - v[1] * v2.v[0]);
	}

	// dot 
	float operator*(vec4& v2) {
		return v[0] * v2.v[0] + v[1] * v2.v[1] + v[2] * v2.v[2];
	}

	float length() {
		return sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2));
	}

	vec4 normalize() {
		return (*this) / (this->length());
	}
};

// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(vec3 image[windowWidth * windowHeight]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		static float vertexCoords[] = { -1, -1, 1, -1, -1, 1,
			1, -1, 1, 1, -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
																							   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

																	  // Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
	}
};

struct Vector {
	const float x, y, z;
	Vector(const float& x, const float& y, const float& z, const float& w = 1)
		:x(x/w), y(y/w), z(z/w) {}

	Vector(const vec4& vec):x(vec.v[0] / vec.v[3]), y(vec.v[1] / vec.v[3]), z(vec.v[2] / vec.v[3]) {}

	const vec4 operator()() const {
		return vec4(x, y, z, 1);
	}

	const Vector operator+(const Vector& v2) const {
		return Vector(x + v2.x, y + v2.y, z + v2.z);
	}

	const Vector operator-(const Vector& v2) const {
		return Vector(x - v2.x, y - v2.y, z - v2.z);
	}

	const Vector operator*(const float& f) const {
		return Vector(x * f, y * f, z * f);
	}

	const Vector operator/(const float& f) const {
		return Vector(x, y, z, f);
	}

	const Vector operator%(const Vector& v2) const {
		return Vector(y*v2.z - z*v2.y, z*v2.x - x*v2.z, x*v2.y - y*v2.x);
	}

	const float operator*(const Vector& v2) const {
		return x*v2.x + y*v2.y + z*v2.z;
	}

	const float length() const {
		return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	}

	const Vector normalize() const {
		if (length()) {
			return (*this) / length();
		}
		else {
			return (*this);
		}
	}
};

struct Material {
	const vec3 color;
	Material(const vec3 color):color(color) {}
};

struct Ray {
	const Vector position;
	const Vector orientation;

	Ray(const Vector& position, const Vector& orientation)
		:position(position), orientation(orientation.normalize()) {}
};


class Light {
protected:
	const vec3 color;

public:
	Light(const vec3& color) :color(color) {}
	virtual const Vector getDirection(const Vector& to) const = 0;
	virtual const float getIntensity(const Vector& to) const = 0;
	virtual const vec3& getColor() const = 0;
};

class LightPoint : public Light {
	const Vector position;
	const float range;
public:
	LightPoint(const Vector& position, const float& range = 300.0, const vec3& color = vec3(1, 1, 1))
		:position(position), Light(color), range(range) {}
	const Vector getDirection(const Vector& to) const {
		return (position - to).normalize();
	}
	const float getIntensity(const Vector& to) const {
		return (pow(range, 2) - pow((position - to).length(),2)) / pow(range, 2);
	}
	const vec3& getColor() const {
		return color;
	}
};

struct Hit {
	const float t;
	const Vector position;
	const Vector normal;
	const Material material;
	Hit(const float& t, const Vector position = vec4(), const Vector normal = vec4(), const Material& material = Material(vec3(1,1,1)))
		:position(position), normal(normal.normalize()), t(t), material(material) {}
	/*Hit(const Hit& hit)
		:position(hit.position), normal(hit.normal), t(hit.t) {}*/
	const vec3 getColor(const std::vector<Light*>& lights) const {
		if (t > 0) {
			vec3 color(0,0,0);
			for (std::vector<Light*>::const_iterator light = lights.begin(); light != lights.end(); light++) {
				color = color + material.color
					*((*light)->getColor())
					*((*light)->getIntensity(position))
					*max(cos((*light)->getDirection(position) * normal), 0.0);
			}
			return color;
		}
		else {
			return vec3(0, 0, 0);
		}
	}
};

class Intersectable {
protected:
	const Material material;
public:
	Intersectable(const Material& material):material(material) {}
	virtual const Hit intersect(const Ray& ray) const = 0;
};

class Plain : public Intersectable {
	const Vector aPoint;
	const Vector normal;
public:
	Plain(const Vector& aPoint, const Vector& normal, const Material& material)
		:aPoint(aPoint), normal(normal.normalize()), Intersectable(material){}

	virtual const Hit intersect(const Ray& ray) const {
		if (ray.orientation*normal != 0) {
			const float t = ((aPoint - ray.position)*normal) / (ray.orientation*normal);
			return Hit(t, ray.position + ray.orientation*t, normal, material);
		}
		else {
			return Hit(-1);
		}
	}
};

struct Camera {
	const Vector position;
	const Vector ahead;
	const Vector up;
	const Vector right;
	FullScreenTexturedQuad texturedQuad;
	vec3* canvas;

	Camera(const Vector& position, const Vector& ahead, const Vector& up,
		const float& horizontalAngle, const float& verticalAngle)
		:position(position), ahead(ahead.normalize()),
		up(up.normalize()*tan(verticalAngle / 2)),
		right((ahead%up).normalize()*tan(horizontalAngle / 2))
	{
		canvas = new vec3[windowHeight*windowWidth];
	}
	
	void render(const std::vector<Intersectable*>& objects,const std::vector<Light*>& lights) {
		for (size_t i = 0; i < windowHeight; i++) {
			if(i%(windowHeight/10)==0)std::cout << (100.0*i)/(windowHeight) << "%" << std::endl;
			for (size_t j = 0; j < windowWidth; j++) {
				const Ray ray(position, ahead + up*(i - windowHeight / 2.0) / (windowHeight / 2.0) + right*(j - windowWidth / 2.0) / (windowWidth / 2.0));
				std::vector<Hit> hits;
				for (std::vector<Intersectable*>::const_iterator object = objects.begin(); object != objects.end(); object++) {
					hits.push_back((*object)->intersect(ray));
				}
				const Hit bestHit = getBestHit(hits);
				canvas[i*windowWidth + j] = bestHit.getColor(lights);
			}
		}
	}

	const Hit& getBestHit(const std::vector<Hit>& hits) const {
		const Hit* bestHit = &hits[0];
		for (std::vector<Hit>::const_iterator hit = hits.begin(); hit != hits.end(); hit++) {
			if ((*hit).t > 0 && (*hit).t > bestHit->t) {
				bestHit = &(*hit);
			}
		}
		return (*bestHit);
	}

	void draw() {
		texturedQuad.Create(canvas);
		texturedQuad.Draw();
	} 

	~Camera() {
		delete[] canvas;
	}
};


Camera* camera;
std::vector<Intersectable*> objects;
std::vector<Light*> lights;
												// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);

	camera = new Camera(
		Vector(0, 0, -300),
		Vector(0, 0, 1),
		Vector(0, 1, 0),
		2, 2
	);
	objects.push_back(new Plain(
		Vector(0, 0, 0),
		Vector(0, 0, 1),
		Material(vec3(1,0,0))
	));
	objects.push_back(new Plain(
		Vector(0, 0, 0),
		Vector(1, 0, 3),
		Material(vec3(0, 1, 0))
	));

	objects.push_back(new Plain(
		Vector(0, 300, 0),
		Vector(0, -1, 4),
		Material(vec3(0, 0, 1))
	));
	lights.push_back(new LightPoint(
		Vector(-10, -200, -10),
		400,
		vec3(0.3, 1, 1)
	));
	lights.push_back(new LightPoint(
		Vector(-50, -50, -50),
		200,
		vec3(1,1,1)
	));
	camera->render(objects, lights);
	camera->draw();
}

void onExit() {
	delete camera;
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	camera->draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	
}

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
