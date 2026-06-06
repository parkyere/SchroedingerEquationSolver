// Humble Object shell -- the GUI / OpenGL boundary.
//
// Per docs/ARCHITECTURE.md this layer contains NO domain logic and is verified
// manually, not by unit tests (the chosen Humble Object pattern). All physics,
// numerics, FFT, math, and geometry generation live in sesolver_core and are
// developed strictly test-first.
//
// For now this is the irreducible minimum: it opens a window hosting an (empty)
// OpenGL viewport, purely to verify the cross-platform Qt + OpenGL build wiring.
// Hand-written OpenGL rendering of the electron cloud is added later (see
// docs/ROADMAP.md, Phase 7).

#include <QApplication>
#include <QMainWindow>
#include <QOpenGLWidget>
#include <QSurfaceFormat>

namespace {

// Empty OpenGL viewport. Rendering logic is intentionally absent at this stage.
class Viewport : public QOpenGLWidget {
public:
    explicit Viewport(QWidget* parent = nullptr) : QOpenGLWidget(parent) {}
};

}  // namespace

int main(int argc, char** argv) {
    // Request a modern OpenGL 4.3 core profile (compute shaders + SSBOs
    // available; see docs/ARCHITECTURE.md for the rationale).
    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication app(argc, argv);

    QMainWindow window;
    window.setWindowTitle(QStringLiteral("Schrodinger Equation Solver"));
    window.setCentralWidget(new Viewport());
    window.resize(1024, 768);
    window.show();

    return app.exec();
}
